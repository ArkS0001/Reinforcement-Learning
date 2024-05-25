import numpy as np

# Define the environment
n_states = 16  # Number of states in the grid world
n_actions = 4  # Number of possible actions (up, down, left, right)
goal_state = 15  # Goal state

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 1000

# Action to state change mapping
action_effects = {
    0: -4,  # Up
    1: 4,   # Down
    2: -1,  # Left
    3: 1    # Right
}

# Function to get the next state based on the current state and action
def get_next_state(current_state, action):
    row, col = divmod(current_state, 4)
    if action == 0 and row > 0:  # Up
        return current_state + action_effects[action]
    elif action == 1 and row < 3:  # Down
        return current_state + action_effects[action]
    elif action == 2 and col > 0:  # Left
        return current_state + action_effects[action]
    elif action == 3 and col < 3:  # Right
        return current_state + action_effects[action]
    return current_state  # If action is invalid, stay in the same state

# SARSA algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)  # Start from a random state
    
    # Choose initial action using epsilon-greedy strategy
    if np.random.rand() < exploration_prob:
        current_action = np.random.randint(0, n_actions)  # Explore
    else:
        current_action = np.argmax(Q_table[current_state])  # Exploit

    while current_state != goal_state:
        # Get the next state based on the current state and action
        next_state = get_next_state(current_state, current_action)

        # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
        reward = 1 if next_state == goal_state else 0

        # Choose next action using epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            next_action = np.random.randint(0, n_actions)  # Explore
        else:
            next_action = np.argmax(Q_table[next_state])  # Exploit

        # Update Q-value using the SARSA update rule
        Q_table[current_state, current_action] += learning_rate * \
            (reward + discount_factor * Q_table[next_state, next_action] - Q_table[current_state, current_action])

        # Move to the next state and action
        current_state = next_state
        current_action = next_action

print("Learned Q-table:")
print(Q_table)
