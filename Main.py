import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# --- Constants ---
EMPTY = 0
WALL = -1
OBSTACLE = 5

ROOMS = {
    'Kitchen': 1,
    'Bedroom': 2,
    'Living Room': 3,
    'Hallway': 4,
}

# --- Create Grid ---
grid_size = 30
room_grid = np.full((grid_size, grid_size), EMPTY)

door_positions = []  # Cache for doors

# --- Helper to Add Room and Walls ---
def add_room(grid, top_left, size, room_id, door_side="right"):
    r, c = top_left
    h, w = size
    grid[r:r+h, c:c+w] = room_id

    # Add walls
    grid[r-1:r, c-1:c+w+1] = WALL
    grid[r+h:r+h+1, c-1:c+w+1] = WALL
    grid[r-1:r+h+1, c-1:c] = WALL
    grid[r-1:r+h+1, c+w:c+w+1] = WALL

    # Add door (clear one wall cell)
    if door_side == "right":
        grid[r + h // 2, c + w] = EMPTY
        door_positions.append((r + h // 2, c + w))
    elif door_side == "left":
        grid[r + h // 2, c - 1] = EMPTY
        door_positions.append((r + h // 2, c - 1))
    elif door_side == "top":
        grid[r - 1, c + w // 2] = EMPTY
        door_positions.append((r - 1, c + w // 2))
    elif door_side == "bottom":
        grid[r + h, c + w // 2] = EMPTY
        door_positions.append((r + h, c + w // 2))

# --- Add Rooms ---
add_room(room_grid, (2, 2), (4, 5), ROOMS['Kitchen'], door_side="right")
add_room(room_grid, (2, 12), (4, 5), ROOMS['Bedroom'], door_side="left")
add_room(room_grid, (10, 2), (5, 6), ROOMS['Living Room'], door_side="top")
add_room(room_grid, (10, 12), (5, 5), ROOMS['Hallway'], door_side="top")

# --- Add Obstacles ---
num_obstacles = 0
for _ in range(num_obstacles):
    while True:
        r = random.randint(0, grid_size - 1)
        c = random.randint(0, grid_size - 1)
        if room_grid[r, c] == EMPTY:
            room_grid[r, c] = OBSTACLE
            break

visual_grid = np.copy(room_grid)
visual_grid[visual_grid == -1] = 6  # Wall remap

# --- Visualization ---
cmap = ListedColormap([
    "white", "lightblue", "lightgreen", "gold", "violet", "red", "black"
])

plt.figure(figsize=(10, 10))
plt.imshow(visual_grid, cmap=cmap)
plt.title("Smart Floor Grid")
plt.grid(True, color='gray', linewidth=0.3)
plt.show()

# --- Q-Learning Setup ---
grid = np.copy(room_grid)
EPISODES = 10000
MAX_STEPS = 100
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

ACTIONS = {
    0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)
}

MOVE_REWARD = -1
GOAL_REWARD = 100
OBSTACLE_REWARD = -10
q_table = np.zeros((grid_size, grid_size, len(ACTIONS)))

while True:
    start = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
    if grid[start] == 0:
        break

def select_random_room_cell(grid, room_name):
    room_id = ROOMS[room_name]
    room_cells = list(zip(*np.where(grid == room_id)))
    return random.choice(room_cells)

goal = select_random_room_cell(room_grid, 'Kitchen')

def is_door_entry(prev_state, new_state):
    return prev_state in door_positions

# --- Training ---
for episode in range(EPISODES):
    state = start
    for step in range(MAX_STEPS):
        x, y = state
        action = random.choice(list(ACTIONS.keys())) if random.random() < EPSILON else np.argmax(q_table[x, y])
        dx, dy = ACTIONS[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            next_state = (new_x, new_y)
            cell_value = grid[new_x][new_y]

            if cell_value in [5, -1, 6]:
                reward = OBSTACLE_REWARD
                next_state = state
            elif (new_x, new_y) == goal:
                reward = GOAL_REWARD
            elif grid[x][y] == EMPTY and cell_value in ROOMS.values():
                if is_door_entry((x, y), (new_x, new_y)):
                    reward = MOVE_REWARD
                else:
                    reward = OBSTACLE_REWARD
                    next_state = state
            else:
                reward = MOVE_REWARD
        else:
            reward = OBSTACLE_REWARD
            next_state = state

        old_value = q_table[x, y, action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        new_value = old_value + LEARNING_RATE * (reward + DISCOUNT * next_max - old_value)
        q_table[x, y, action] = new_value

        state = next_state
        if state == goal:
            break

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
# --- Extract Path ---
def extract_path(start, goal):
    path = [start]
    current = start
    for _ in range(100):
        x, y = current
        action = np.argmax(q_table[x, y])
        dx, dy = ACTIONS[action]
        next_pos = (x + dx, y + dy)
        if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
            break
        if grid[next_pos] in [-1, 5, 6] or next_pos in path:
            break
        path.append(next_pos)
        current = next_pos
        if current == goal:
            break
    return path

# --- Visualization ---
def visualize_path(path):
    visual = np.copy(visual_grid)
    for (x, y) in path:
        visual[x][y] = 10
    sx, sy = start
    gx, gy = goal
    visual[sx][sy] = 8
    visual[gx][gy] = 8
    cmap = ListedColormap([
        "white", "lightblue", "lightgreen", "gold", "violet",
        "red", "black", "blue", "green", "lime", "cyan"])
    plt.figure(figsize=(10, 10))
    plt.imshow(visual, cmap=cmap)
    plt.title("Q-Learning Path Visualization")
    plt.grid(True, color='gray', linewidth=0.3)
    plt.show()

path = extract_path(start, goal)
visualize_path(path)
print("Path:", path)
