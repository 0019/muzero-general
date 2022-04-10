import datetime
import pathlib
import cv2
import numpy
import torch
numpy.set_printoptions(threshold=numpy.inf)

from supersuit import frame_stack_v1, color_reduction_v0, \
    max_observation_v0, frame_skip_v0, resize_v0, clip_reward_v0, black_death_v3
from pettingzoo.atari import boxing_v1

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 10  # Seed for numpy, torch and the game
        self.max_num_gpus = 1  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (4, 56, 50)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(18))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 2000  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [32]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [32]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 0.999  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = self.preset_settings_env(boxing_v1.env())
        if seed is not None:
            self.env.seed(seed)
        self.scores = {}
        self.steps = 0
        self.step_limit = 1500
        self.reset_game()
        # self.env = gym.make("ALE/Boxing-v5")

    def reset_game(self):
        self.steps = 0
        self.scores["first_0"] = 0
        self.scores["second_0"] = 0

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        action_agent = self.env.agent_selection
        self.env.step(action)
        self.scores[action_agent] += self.env.rewards[action_agent]
        self.steps += 1

        done = False
        reward = 0
        if self.steps >= self.step_limit:
            if self.scores["first_0"] == self.scores["second_0"]:
                done = True
            if action_agent == "first_0" and self.scores["first_0"] > self.scores["second_0"]:
                done = True
                reward = self.scores["first_0"] - self.scores["second_0"]
            elif action_agent == "second_0" and self.scores["second_0"] > self.scores["first_0"]:
                done = True
                reward = self.scores["second_0"] - self.scores["first_0"]

        # crop image
        a = numpy.moveaxis(self.env.observe(action_agent), -1, 0)
        a = self.crop_and_color_image(a)
        if action_agent == "second_0":
            a = self.flip_image_color(a)

        return a, reward * 10, done

    def crop_and_color_image(self, a):
        a = a[:, :, 17:67]
        a = a[:, 15:71, :]
        # black set to 10, white set to 255
        a = numpy.where(a >= 120, 255, a)  # to full white
        a = numpy.where(a <= 100, 10, a)  # to full black
        return a

    def flip_image_color(self, a):
        a = numpy.where(a == 255, 1, a)  # white to black
        a = numpy.where(a == 10, 255, a)  # black to white
        a = numpy.where(a == 1, 10, a)  # black to white
        return a

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(18))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        self.env.reset()
        self.reset_game()
        # move agents closer to each other
        for i in range(8):
            self.env.step(8)
            self.env.step(7)
        observation, _, _, _ = self.env.last()
        a = numpy.moveaxis(observation, -1, 0)
        a = self.crop_and_color_image(a)
        return a

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        # input("Press enter to take a step ")

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        if self.env.agent_selection == "first_0":
            return 0
        return 1

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        # actions = {
        #     0: "Push cart to the left",
        #     1: "Push cart to the right",
        # }
        return f"{action_number}. {action_number}"

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(
            f"Enter action for the player {self.to_play()}: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter action: ")
        return int(choice)

    def preset_settings_env(self, env):
        '''
        Presets certain conditions on the environment for
        better scaling of the observations etc.
        '''
        ## doing the max and skip as defined by deepmind.
        # env = max_observation_v0(env, memory=2)
        env = frame_skip_v0(env, 4)
        env = black_death_v3(env)

        env = frame_stack_v1(color_reduction_v0(env, 'R'), 4)

        env = resize_v0(env, 84, 84, linear_interp=True)

        # env = clip_reward_v0(env, -1, 1)

        return env
