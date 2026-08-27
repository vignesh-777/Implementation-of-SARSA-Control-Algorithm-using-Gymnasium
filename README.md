# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement



## Software Requirements



## Environment Description



## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm


## Python Program

```python
import numpy as np
import matplotlib.pyplot as plt

# =================================================
# Custom Environment
# =================================================

grid = [
    "FFFS",
    "FHFF",
    "FFHF",
    "GFFF"
]
start_state = 3
goal_state = 12
holes = [5, 10]

num_states = 16
num_actions = 4


# =================================================
# Hyperparameters
# =================================================

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995
# =================================================
# Q-table
# =================================================

Q = np.zeros((num_states, num_actions))

episode_rewards = []


# =================================================
# Epsilon-Greedy
# =================================================

def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        return np.random.randint(num_actions)

    max_q = np.max(Q[state])

    best_actions = np.flatnonzero(
        Q[state] == max_q
    )

    return np.random.choice(best_actions)
# =================================================
# Environment Step
# =================================================

def step(state, action):

    row = state // 4
    col = state % 4

    # 0 = Left
    # 1 = Down
    # 2 = Right
    # 3 = Up

    if action == 0:
        col -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        col += 1
    elif action == 3:
        row -= 1

    # Keep inside grid
    row = max(0, min(3, row))
    col = max(0, min(3, col))

    next_state = row * 4 + col

    # Goal
    if next_state == goal_state:
        return next_state, 1.0, True

    # Hole
    if next_state in holes:
        return next_state, -1.0, True

    # Normal movement
    return next_state, -0.01, False

# =================================================
# SARSA Training
# =================================================

for episode in range(num_episodes):

    state = start_state

    action = epsilon_greedy_action(
        state,
        epsilon
    )

    total_reward = 0

    for step_count in range(max_steps_per_episode):

        next_state, reward, done = step(
            state,
            action
        )

        total_reward += reward

        # Terminal state
        if done:

            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

            break

        # Choose next action
        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # SARSA update
        Q[state, action] += alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        state = next_state
        action = next_action

    episode_rewards.append(total_reward)

    # Epsilon decay
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# =================================================
# State-Value Function
# =================================================

state_values = np.max(Q, axis=1)


# =================================================
# Learned Policy
# =================================================

learned_policy = np.argmax(Q, axis=1)


# =================================================
# Output
# =================================================

print("\nCustom Environment:")

for row in grid:
    print(" ".join(row))


print("\nFinal Q-table:")
print(np.round(Q, 3))


print("\nEstimated State-Value Function:")
print(
    np.round(
        state_values.reshape(4, 4),
        3
    )
)


# =================================================
# Policy Display
# =================================================

action_symbols = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

print("\nLearned Policy:")

for state in range(num_states):

    if state == start_state:
        symbol = "S"

    elif state == goal_state:
        symbol = "G"

    elif state in holes:
        symbol = "H"

    else:
        symbol = action_symbols[
            learned_policy[state]
        ]

    print(symbol, end=" ")

    if (state + 1) % 4 == 0:
        print()


# =================================================
# Average Reward
# =================================================

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 3)
)


# =================================================
# Learning Curve
# =================================================

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "SARSA Learning Curve - Custom Environment"
)

plt.grid(True)
plt.tight_layout()
plt.show()


```
---

## Output

Final Q-table:

<img width="319" height="332" alt="image" src="https://github.com/user-attachments/assets/a9b3e8b2-3dcb-4852-8bd3-3047b7cad6d6" />





Estimated State-Value Function:

<img width="296" height="115" alt="image" src="https://github.com/user-attachments/assets/695e45ad-8b8e-49ee-9573-b0faf7f1a716" />




Learned Policy:

<img width="226" height="104" alt="image" src="https://github.com/user-attachments/assets/a907c0b3-bb5b-49bc-929f-ee7b85bcf23d" />



Average reward over last 1000 episodes:

<img width="759" height="476" alt="image" src="https://github.com/user-attachments/assets/b8d834a9-b6b1-4f47-a97d-156567349963" />



---

## Result


---

## Inference
```text



```
---

