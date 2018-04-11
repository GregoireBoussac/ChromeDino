import numpy as np


class Training():
    def __init__(self):
        # hyperparameters
        self.episode_number = 0
        self.batch_size = 5
        self.gamma = 0.99 # discount factor for reward
        self.decay_rate = 0.99
        self.num_hidden_layer_neurons = 200
        self.input_dimensions = 100 * 600
        self.learning_rate = 1e-4
        self.running_reward = 0
        
        self.memory_rewards = []

        # weights of the 2 layers
        self.weights = {
          '1': np.random.randn(self.num_hidden_layer_neurons, self.input_dimensions) /np.sqrt(self.input_dimensions),
          '2': np.random.randn(self.num_hidden_layer_neurons) / np.sqrt(self.num_hidden_layer_neurons)
        }

        # parameters for RMSProp
        # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
        self.expectation_g_squared = {}
        self.g_dict = {}
        for layer_name in self.weights.keys():
            self.expectation_g_squared[layer_name] = np.zeros_like(self.weights[layer_name])
            self.g_dict[layer_name] = np.zeros_like(self.weights[layer_name])
            
            
        for episode_number in range(80):
            print('>> EPISODE ', episode_number)
            
            window_name = "Game " + str(episode_number)
            game = Game(self, window_name, True)
            
            self.memory_rewards.append(game.final_score)
            reward_sum = game.final_score
            
            # The next step is to figure out how we learn after the end of an episode (i.e. game over).
            # We do this by computing the policy gradient of the network at the end of each episode.
            # The intuition here is that if we won, we’d like our network to generate more of the actions that led to us winning.
            # Alternatively, if we lose, we’re going to try and generate less of these actions.
            # When we notice we are done, the first thing we do is compile all our observations and gradient calculations for the episode.
            # This allows us to apply our learnings over all the actions in the episode.
            
            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(game.episode_hidden_layer_values)
            episode_observations = np.vstack(game.episode_observations)
            episode_gradient_log_ps = np.vstack(game.episode_gradient_log_ps)
            episode_rewards = np.vstack(game.episode_rewards)
            print('game.episode_rewards: ', game.episode_rewards)

            # we want to learn in such a way that actions taken towards the end of an episode more heavily influence our learning than actions taken at the beginning.
            # This is called discounting.
            # Tweak the gradient of the log_ps based on the discounted rewards
            # After this, we’re going to finally use backpropagation to compute the gradient
            # (i.e. the direction we need to move our weights to improve).
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, self.gamma)
            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                self.weights
            )

            # batch_size : number of episodes before updating the weights of the NN
            if episode_number % self.batch_size == 0:
                update_weights(self.weights, self.expectation_g_squared, self.g_dict, self.decay_rate, self.learning_rate)

            #self.episode_hidden_layer_values, self.episode_observations, self.episode_gradient_log_ps, self.episode_rewards = [], [], [], [] # reset values
            # observation = env.reset() # reset env
            self.running_reward = 0.01*reward_sum + self.running_reward * 0.99
            print ('resetting env. episode {} reward total was running mean: {}'.format(reward_sum, self.running_reward))