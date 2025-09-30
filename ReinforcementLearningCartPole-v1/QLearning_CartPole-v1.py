import gym 
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input  # Import Input layer
from tensorflow.keras.optimizers import Adam
import gym  # Ensure the environment library is available
import random
import numpy as np
from collections import deque
import tensorflow as tf

env = gym.make('CartPole-v1')

episodes = 10  
batch_size = 32  
memory = deque(maxlen=2000)  

state_size = env.observation_space.shape[0]  
action_size = env.action_space.n 

# Define the model
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Input(shape=(state_size,)))  
    model.add(Dense(32, activation='relu'))  
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Re-initialize the model with the new architecture
model = build_model(state_size, action_size)

def act(state):
    return env.action_space.sample()  # For now, a random action is taken

# Function to remember experiences in memory
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Optimized function to replay experiences from memory and train the model
def replay(batch_size):
    minibatch = random.sample(memory, batch_size)
    states = np.vstack([sample[0] for sample in minibatch])
    next_states = np.vstack([sample[3] for sample in minibatch])
    targets = model.predict(states)
    target_next = model.predict(next_states)
    
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        target = reward if done else reward + 0.95 * np.amax(target_next[i])
        targets[i][action] = target
        
    model.fit(states, targets, epochs=1, verbose=0)

# Training the model
for e in range(episodes):
    state, _ = env.reset()  
    state = np.reshape(state, [1, state_size])
    for time in range(200):  
        action = act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            print(f"episode: {e+1}/{episodes}, score: {time}")
            break
        
        if len(memory) > batch_size and time % 10 == 0: 
            replay(batch_size)  

env.close()