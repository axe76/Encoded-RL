import numpy as np
import keras
from keras.layers import Dense, Flatten
from keras.models import load_model
import os
import numpy as np
import argparse
from tensorflow.python.client import device_lib
import argparse
import keras
from keras.layers import Dense, Flatten
import gym
import time
import warnings

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(device_lib.list_local_devices())
warnings.filterwarnings("ignore") # Ignores all warning messages
keras.backend.set_image_data_format('channels_last')

def preprocess(image):
    image = image[35:195:2, ::2, 0]
    image[np.logical_or(image == 144, image == 109)] = 0
    image[image != 0] = 1
    image = image.astype(np.float)
    image = image.astype('float32') / 255
    image = np.reshape(image, (1, 80, 80, 1)) 
    return image

def choose_mode(v):
    if v.lower() in ('Train','train'):
        return 'Train'
    elif v.lower() in ('Test','test'):
        return 'Test'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def discounted_rewards(rew, gamma=0.99):
    d_r = np.zeros_like(rew)
    p_r = 0
    for t in range(rew.size - 1, 0, -1):
        if rew[t] != 0: p_r = 0
        p_r = p_r * gamma + rew[t]
        d_r[t] = p_r
    d_r -= np.mean(d_r)
    d_r /= np.std(d_r)
    return d_r

def build_dqn_model():
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=((10, 10, 8))))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    if os.path.isfile('Pong_agent'):
        model.load_weights('Pong_agent')
    return model

encoder = load_model('./weights/encoder_weights.h5',compile=False)

def train_agent():
    # render = True
    render = False
    lr = 0.1
    states = []
    action_probs = []
    action_prob_grads = []
    rewards = []
    reward_sum = 0
    reward_sums = []
    episode_counter = 0

    env = gym.make("Pong-v0")
    observation = env.reset()
    previous_observation = preprocess(observation)
    previous_observation = encoder.predict(previous_observation)
    agent = build_dqn_model()

    while True:
        if render:
            env.render()
        current_observation = preprocess(observation)
        current_observation = encoder.predict(current_observation)
        state = current_observation - previous_observation
        previous_observation = current_observation

        action_prob = agent.predict_on_batch(state.reshape(1, 10, 10, 8))[0,:]
        action = np.random.choice(6, p=action_prob)

        observation, reward, done, _ = env.step(action)
        reward_sum += reward

        states.append(state)
        action_probs.append(action_prob)
        rewards.append(reward)
        y = np.zeros(6)
        y[action] = 1
        action_prob_grads.append(y - action_prob)
    
        if done:
            # Game Over - One of the players has gotten 21 points
            episode_counter += 1
            reward_sums.append(reward_sum)
            if len(reward_sums) > 40:
                reward_sums.pop(0)

            print('Episode: %d ------- Total Episode Reward: %f ------- Mean %f' % (episode_counter, reward_sum, np.mean(reward_sums)))
                
            rewards = np.vstack(rewards)
            action_prob_grads = np.vstack(action_prob_grads)
            rewards = discounted_rewards(rewards)

            X = np.vstack(states).reshape(-1, 10, 10, 8)
            Y = action_probs + lr * rewards * action_prob_grads
            
            agent.train_on_batch(X, Y)

            agent.save_weights('Pong_agent')
            
            states, action_prob_grads, rewards, action_probs = [], [], [], []
            reward_sum = 0
            observation = env.reset()

def test_agent():
    env = gym.make("Pong-v0")
    observation = env.reset()
    previous_observation = preprocess(observation)
    previous_observation = encoder.predict(previous_observation)
    agent = model.build_dqn_model()

    while True:
        env.render()
        current_observation = preprocess(observation)
        current_observation = encoder.predict(current_observation)
        state = current_observation - previous_observation
        previous_observation = current_observation
        action_prob = agent.predict_on_batch(state.reshape(1, 80, 80, 1))[0,:]
        action = np.random.choice(6, p=action_prob)
        observation, reward, done, _ = env.step(action)
        time.sleep(1/60)  
        if done:
            observation = env.reset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=choose_mode, nargs='?',const=True,help="Train from scratch or Test Pre-Trained Agent")
    args = parser.parse_args()
    mode = args.mode
    if mode == 'Train':
        t1 = time.time()
        train_agent()
        t2 = time.time()
        print('Time Taken for Training : ',t2-t1)
    else:
        test_agent()

main()