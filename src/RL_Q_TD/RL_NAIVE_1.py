import asyncio
import websockets
import json
import numpy as np

print("Loading RL server...")

# ------------------------------------------------
# ACTION SPACE
# ------------------------------------------------
# Movement types (these do NOT contain dx/dy)
MOVE_ACTIONS = [
    "PURSUIT_FOOD",
    "RUN_FROM_PREDATOR",
]

FOOD_ACTIONS = ["REPRODUCE", "ACTIVATE_SPECIAL", "IDLE_FOOD"]

ACTION_COUNT = len(MOVE_ACTIONS) * len(FOOD_ACTIONS)


def decode_action(idx):
    move_idx = idx // len(FOOD_ACTIONS)
    food_idx = idx % len(FOOD_ACTIONS)
    return MOVE_ACTIONS[move_idx], FOOD_ACTIONS[food_idx]


# ------------------------------------------------
# Q-Learning parameters
# ------------------------------------------------
EPSILON = 0.08
ALPHA = 0.25
GAMMA = 0.92

Q = {}   # { preyId → { state_key → np.array } }


def ensure_Q(prey_id, state_key):
    """Ensure Q-table entry exists."""
    if prey_id not in Q:
        Q[prey_id] = {}
    table = Q[prey_id]
    if state_key not in table:
        table[state_key] = np.zeros(ACTION_COUNT)
    print("Q-table for prey", prey_id, ":", table)
    return table[state_key]


# ------------------------------------------------
# Convert JS state → key
# ------------------------------------------------
def state_to_key(state):
    # Example state format:
    # [dx_food, dy_food, pred_dx,y_or_null, starving_flag]
    cleaned = []
    for v in state:
        if isinstance(v, (int, float)):
            cleaned.append(round(float(v), 3))
        else:
            cleaned.append(v)
    # print("cleaned state",cleaned)
    return str(tuple(cleaned))


# ------------------------------------------------
# Epsilon-greedy action selection
# ------------------------------------------------
def choose_action(prey_id, state_key):
    q = ensure_Q(prey_id, state_key)
    if np.random.rand() < EPSILON:
        return np.random.randint(ACTION_COUNT)
    return int(np.argmax(q))


# ------------------------------------------------
# Q-learning update
# ------------------------------------------------
def update_Q(prey_id, prev_key, action_idx, reward, state_key):
    q_prev = ensure_Q(prey_id, prev_key)
    q_next = ensure_Q(prey_id, state_key)

    td_target = reward + GAMMA * np.max(q_next)
    td_error = td_target - q_prev[action_idx]

    q_prev[action_idx] += ALPHA * td_error


# ------------------------------------------------
# Compute dx/dy from state and move_action
# ------------------------------------------------
def movement_from_state(raw_state, move_action):
    dx_food, dy_food, dx_pred, dy_pred, prednearby, starving = raw_state

    # Pursue food
    if move_action == "PURSUIT_FOOD":
        return dx_food, dy_food

    # Run from predator if exists
    if move_action == "RUN_FROM_PREDATOR" :
        return dx_pred,dy_pred

# ------------------------------------------------
# WebSocket handler
# ------------------------------------------------
async def handler(websocket):
    print("Client connected")

    async for message in websocket:
        data = json.loads(message)

        prey_id = data["preyId"]
        raw_state = data["state"]
        reward = float(data.get("reward", 0))

        state_key = state_to_key(raw_state)

        # Q-update if previous step exists
        prev_state = data.get("prev_state")
        prev_action = data.get("prev_action")

        if prev_state is not None and prev_action is not None:
            prev_key = state_to_key(prev_state)
            update_Q(prey_id, prev_key, prev_action, reward, state_key)

        # Choose next action
        action_idx = choose_action(prey_id, state_key)
        move_action, food_action = decode_action(action_idx)

        # Compute dx, dy from current state
        dx, dy = movement_from_state(raw_state, move_action)

        # Send back result
        await websocket.send(json.dumps({
            "preyId": prey_id,
            "dx": dx,
            "dy": dy,
            "action_idx": action_idx,
            "moveAction": move_action,
            "foodAction": food_action
        }))


# ------------------------------------------------
# Main
# ------------------------------------------------
async def main():
    print("Python RL server running on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
