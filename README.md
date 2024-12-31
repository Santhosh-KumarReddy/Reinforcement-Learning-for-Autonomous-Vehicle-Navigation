# Reinforcement-Learning-for-Autonomous-Vehicle-Navigation
Implemented deep reinforcement learning algorithms (DDQN, DQN) to optimize autonomous vehicle navigation, improving course completion times by 15% and evaluating generalization through transfer learning in simulated 2D environments

# Visual Simulation of Autonomous Vehicle Navigation

![Visual Simulation](https://media.giphy.com/media/6yGw51hg6gRflukHz8/giphy.gif)

This is a visual simulation of the autonomous vehicle navigating through the environment.


## Termination Conditions

The episode terminates under the following conditions:

1. **Collision with Borders**  
   - The car colliding with the environment's borders results in failure and a large negative reward.  
   - **Penalty**: `-1000`.

2. **Touching the Start Color**  
   - If the car touches the blue starting area, the episode ends with a penalty applied.  
   - **Penalty**: `-500`.

3. **Collision with Stationary Cars**  
   - Collisions with stationary vehicles lead to immediate termination and a significant penalty.  
   - **Penalty**: `-1000`.

4. **Reaching the Goal**  
   - Successfully reaching the red goal area terminates the episode with a large positive reward.  
   - **Reward**: `1000 - Time Taken`.

---

## Reward System

The reward system is designed to encourage efficient navigation and discourage mistakes:

1. **Goal Reward**  
   - A large positive reward of `1000 - Time Taken` is awarded for reaching the goal.  
   - This incentivizes the agent to complete the task as quickly as possible.

2. **Collision Penalties**  
   - Collisions with borders or stationary cars result in a penalty of `-1000`.  
   - Returning to the initial position results in a penalty of `-500`.

3. **Time Penalty**  
   - A small penalty of `-1` per time step encourages the agent to find the most efficient path.

---

## Transfer Learning

The project explores **transfer learning** to enhance the agent’s generalization capabilities:

- **Saved Model Performance**  
   The trained model's performance in one environment is saved.

- **New Environment Adaptation**  
   The saved model is deployed in a new environment, requiring less exploration while retaining high performance.

By leveraging transfer learning, the agent adapts more quickly to new scenarios, demonstrating robust navigation abilities.

---
