import math
import random
import pygame
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # Added for plotting

WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60
CAR_SIZE_Y = 50
BORDER_COLOR = (255, 255, 255, 255)
GOAL_COLOR = (255, 0, 0, 255)  # Red color for the goal
START_COLOR = (0, 0, 255, 255)  # Blue color for the start
STATIONARY_CAR_COLOR = (0, 255, 0, 255)  # Green color for stationary cars
MAX_SPEED = 25  # Define a maximum speed

class CarEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Car Simulation")
        self.clock = pygame.time.Clock()
        self.car = pygame.image.load('car.png').convert_alpha()
        self.car = pygame.transform.scale(self.car, (CAR_SIZE_X, CAR_SIZE_Y))
        self.map = pygame.image.load('main3.png').convert()
        self.font = pygame.font.SysFont('Arial', 25)
        self.reset()

    def reset(self):
        self.car_pos = [850, 925]
        self.car_angle = 0
        self.speed = 0
        self.alive = True
        self.distance = 0
        self.time = 0
        self.radars = []
        self.update_radar()
        return self.get_state()

    def step(self, action):
        if action == 0:
            self.car_angle += 10  # Turn left
        elif action == 1:
            self.car_angle -= 10  # Turn right
        elif action == 2:
            self.speed = max(self.speed - 2, 8)  # Slow down with minimum speed constraint
        else:
            self.speed = min(self.speed + 2, MAX_SPEED)  # Speed up with maximum speed constraint

        self.move()
        self.check_collision()
        self.update_radar()
        reward = self.get_reward()
        state = self.get_state()
        done = not self.alive or self.check_goal() or self.check_start_color()
        return state, reward, done, {}

    def move(self):
        self.car_pos[0] += math.cos(math.radians(360 - self.car_angle)) * self.speed
        self.car_pos[1] += math.sin(math.radians(360 - self.car_angle)) * self.speed
        self.distance += self.speed
        if self.speed > 0:  # Only increment time if speed is greater than 0
            self.time += 1

    def check_collision(self):
        self.alive = True
        for point in self.corners:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                color = self.map.get_at((x, y))
                if color == BORDER_COLOR:
                    self.alive = False
                    self.collision_penalty = -1000  # Penalty for collision with border
                    break
                elif color == STATIONARY_CAR_COLOR:
                    self.alive = False
                    self.collision_penalty = -1000  # Penalty for collision with stationary car
                    break
            else:
                self.alive = False  # If the point is out of bounds, consider it a collision with border
                self.collision_penalty = -1000
                break

    def check_goal(self):
        for point in self.corners:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if self.map.get_at((x, y)) == GOAL_COLOR:
                    return True
        return False

    def check_start_color(self):
        for point in [self.corners[0], self.corners[1]]:  # Check front two corners
            x, y = int(point[0]), int(point[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if self.map.get_at((x, y)) == START_COLOR:
                    self.alive = False  # Terminate the episode
                    return True
        return False

    def update_radar(self):
        self.corners = self.calculate_corners()
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d)

    def check_radar(self, degree):
        length = 0
        x = int(self.car_pos[0] + CAR_SIZE_X / 2 + math.cos(math.radians(360 - (self.car_angle + degree))) * length)
        y = int(self.car_pos[1] + CAR_SIZE_Y / 2 + math.sin(math.radians(360 - (self.car_angle + degree))) * length)
        while not self.map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.car_pos[0] + CAR_SIZE_X / 2 + math.cos(math.radians(360 - (self.car_angle + degree))) * length)
            y = int(self.car_pos[1] + CAR_SIZE_Y / 2 + math.sin(math.radians(360 - (self.car_angle + degree))) * length)
        dist = int(math.sqrt(math.pow(x - (self.car_pos[0] + CAR_SIZE_X / 2), 2) + math.pow(y - (self.car_pos[1] + CAR_SIZE_Y / 2), 2)))
        self.radars.append((x, y, dist))

    def calculate_corners(self):
        length = 0.5 * CAR_SIZE_X
        center = [self.car_pos[0] + CAR_SIZE_X / 2, self.car_pos[1] + CAR_SIZE_Y / 2]
        left_top = [center[0] + math.cos(math.radians(360 - (self.car_angle + 30))) * length, center[1] + math.sin(math.radians(360 - (self.car_angle + 30))) * length]
        right_top = [center[0] + math.cos(math.radians(360 - (self.car_angle + 150))) * length, center[1] + math.sin(math.radians(360 - (self.car_angle + 150))) * length]
        left_bottom = [center[0] + math.cos(math.radians(360 - (self.car_angle + 210))) * length, center[1] + math.sin(math.radians(360 - (self.car_angle + 210))) * length]
        right_bottom = [center[0] + math.cos(math.radians(360 - (self.car_angle + 330))) * length, center[1] + math.sin(math.radians(360 - (self.car_angle + 330))) * length]
        return [left_top, right_top, left_bottom, right_bottom]

    def get_state(self):
        radar_distances = [dist for _, _, dist in self.radars]
        return np.array(radar_distances + [self.speed])

    def get_reward(self):
        if self.check_goal():
            return 1000 - self.time  # Large reward for reaching the goal quickly
        if not self.alive:
            return self.collision_penalty  # Apply the specific penalty for the type of collision
        if self.check_start_color():
            return -500  # Penalty for touching blue color
        return -1  # Penalty for each time step to encourage faster completion

    def render(self, episode):
        self.screen.blit(self.map, (0, 0))
        rotated_car = pygame.transform.rotate(self.car, self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_pos[0] + CAR_SIZE_X / 2, self.car_pos[1] + CAR_SIZE_Y / 2))
        self.screen.blit(rotated_car, car_rect.topleft)

        for radar in self.radars:
            x, y, _ = radar
            pygame.draw.line(self.screen, (0, 255, 0), (int(self.car_pos[0] + CAR_SIZE_X / 2), int(self.car_pos[1] + CAR_SIZE_Y / 2)), (x, y), 1)
            pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 5)

        episode_text = self.font.render(f"Episode: {episode}", True, (0, 0, 0))
        distance_text = self.font.render(f"Distance: {self.distance:.1f}", True, (0, 0, 0))
        speed_text = self.font.render(f"Speed: {self.speed:.1f}", True, (0, 0, 0))
        time_text = self.font.render(f"Time: {self.time}", True, (0, 0, 0))
        self.screen.blit(episode_text, (10, 10))
        self.screen.blit(distance_text, (10, 40))
        self.screen.blit(speed_text, (10, 70))
        self.screen.blit(time_text, (10, 100))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDQNAgent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.losses = []  # Track losses

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, current_episode):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                next_action = torch.argmax(self.model(next_state)).item()
                target = reward + self.gamma * self.target_model(next_state)[0][next_action].item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min and current_episode < stop_exploring_episode:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

if __name__ == "__main__":
    env = CarEnv()
    state_size = env.reset().shape[0]
    action_size = 4
    agent = DDQNAgent(state_size, action_size)
    episodes = 1000
    episode_rewards = []  # Track rewards
    average_speeds = []  # Track average speeds
    stop_exploring_episode = 900

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_speed = 0
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if env.speed > 0:
                total_speed += env.speed
                steps += 1
            env.render(e + 1)

        episode_rewards.append(total_reward)
        if steps > 0:
            average_speed = total_speed / steps
            average_speeds.append(average_speed)
        else:
            average_speeds.append(0)

        agent.replay(e)  # Pass the current episode
        if e % 10 == 0:
            agent.update_target_model()

    torch.save(agent.model.state_dict(), 'ddqn_model.pth')
    torch.save(agent.target_model.state_dict(), 'dqn_target_model.pth')

    env.close()

    # Plot Loss Over Time
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(agent.losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')

    plt.subplot(1, 3, 2)
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')

    plt.subplot(1, 3, 3)
    plt.plot(average_speeds)
    plt.xlabel('Episodes')
    plt.ylabel('Average Speed')
    plt.title('Average Speed Per Episode')

    plt.tight_layout()
    plt.show()

