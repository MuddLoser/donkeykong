import gym
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import moviepy
from collections import deque
import os
from gym.wrappers import RecordVideo

os.environ['SDL_VIDEODRIVER'] = 'dummy'
env = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array')
#env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: x % 50 == 0)

action_size = env.action_space.n
num_episodes = 300
learning_rate = 0.00025
discount_rate = 0.99
epsilon = 0.95
epsilon_min = 0.1
epsilon_decay = epsilon_decay = (epsilon - epsilon_min) / num_episodes
batch_size = 32
num_steps = 300
memory_size = 100_000
target_update_freq = 100
replay_freq = 4

def preprocess_frame(frame): 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0 

class CNN_DQN(nn.Module):
    def __init__(self, action_size):
        super(CNN_DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / num_episodes
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CNN_DQN(action_size).to(self.device)
        self.target_model = CNN_DQN(action_size).to(self.device)
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.from_numpy(np.stack(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        dones = torch.tensor(dones).float().unsqueeze(1).to(self.device)

        Q_expected = self.model(states).gather(1, actions)
        Q_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + (self.gamma * Q_next * (1 - dones))

        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()

# Entrenamiento
agent = DQNAgent(action_size)
step_count = 0
rewards_history = []

for episode in range(num_episodes):
    state_raw, _ = env.reset()
    state = preprocess_frame(state_raw)
    
    frame_stack = deque([state] * 4, maxlen=4)
    state_stack = np.stack(frame_stack, axis=0)

    total_reward = 0

    for step in range(num_steps):
        action = agent.act(state_stack)
        next_state_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = preprocess_frame(next_state_raw)

        frame_stack.append(next_state)
        next_state_stack = np.stack(frame_stack, axis=0)

        clipped_reward = np.clip(reward, -1, 1)
        agent.remember(state_stack, action, clipped_reward, next_state_stack, done)

        state_stack = next_state_stack
        state = next_state
        total_reward += reward
        step_count += 1

        if step_count % replay_freq == 0 and len(agent.memory) > agent.batch_size * 2:
            agent.replay()

        if done:
            break

    rewards_history.append(total_reward)
    
    if episode % target_update_freq == 0:
        agent.update_target_model()

    avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
    print(f"Episode {episode + 1}/{num_episodes} - Score: {total_reward:.2f} - Avg50: {avg_reward:.2f} - Epsilon: {agent.epsilon:.4f} - Memory: {len(agent.memory)}")

    if episode > 200 and agent.epsilon < 0.3:
        replay_freq = 8

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)


env.close()
print("Entrenamiento terminado.")

try:
    from gym.wrappers import RecordVideo

    video_env = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array')
    video_env = RecordVideo(video_env, video_folder='./videos', episode_trigger=lambda episode_id: True)

    state_raw, _ = video_env.reset()
    state = preprocess_frame(state_raw)
    frame_stack = deque([state] * 4, maxlen=4)
    state_stack = np.stack(frame_stack, axis=0)

    done = False
    while not done:
        action = agent.act(state_stack)
        next_state_raw, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated

        next_state = preprocess_frame(next_state_raw)
        frame_stack.append(next_state)
        state_stack = np.stack(frame_stack, axis=0)

    video_env.close()
    print("Episodio grabado.")

except ImportError:
    print("'moviepy' no está instalado. Ejecuta: pip install moviepy")
except Exception as e:
    print("Ocurrió un error al grabar el video:", str(e))
 