import gym
import numpy as np
import matplotlib.pyplot as plt

num_images = 100
env = gym.make('Pong-v0')
env.reset()
done = False
path = 'env_images_test/'
for i in range(1000):
    # env.render()
    state, reward, done, _ = env.step(env.action_space.sample())
    state = state[35:195:2, ::2, 0]
    state[np.logical_or(state == 144, state == 109)] = 0
    state[state != 0] = 1
    state = state.astype(np.float)
    # state = np.expand_dims(state, axis=-1)
    if i>=500 and i<=500+num_images:
        plt.imsave(path+'{}.png'.format(i), state, cmap='gray')
    # print(state.shape)