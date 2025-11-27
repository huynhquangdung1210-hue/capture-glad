import numpy as np
import websockets, asyncio
import glob
import json
import random
import math

EPISODE = 72

# ------------------------------------------------
# ACTION SPACE
# ------------------------------------------------
# Movement types (these do NOT contain dx/dy)
MOVE_ACTIONS = [
    "PURSUIT_FOOD",
    "RUN_FROM_PREDATOR",
    "RUN_TO_PREDATOR",
]

FOOD_ACTIONS = ["REPRODUCE", "ACTIVATE_SPECIAL", "IDLE_FOOD"]

ACTION_COUNT = len(MOVE_ACTIONS) * len(FOOD_ACTIONS)


def decode_action(idx):
    move_idx = idx // len(FOOD_ACTIONS)
    food_idx = idx % len(FOOD_ACTIONS)
    return MOVE_ACTIONS[move_idx], FOOD_ACTIONS[food_idx]


# ------------------------------------------------
# Convert JS state → key
# ------------------------------------------------
global predatornearbycutoff
predatornearbycutoff=100
def state_to_key(state):
    """Convert a raw state list into a compact, hashable string key.

    Rounds numeric values and converts booleans to ints to keep keys small
    and consistent between client and backend simulator.
    """
    [prey,sorted_stars,sorted_predators]=state
    if state is None:
        return "None"
    
    cleaned = []
    dist_nearestpred=math.sqrt((prey['x']-sorted_predators[0]['x'])**2+(prey['y']-sorted_predators[0]['y'])**2) if sorted_predators else 300
    prednearby= int(dist_nearestpred<predatornearbycutoff)
    preyenergy=prey['energy']
    dist_prey_center = round(float(math.sqrt(prey["x"]**2 + prey["y"]**2)), -2)
    cleaned=[preyenergy,prednearby,dist_prey_center]
    return str(tuple(cleaned))

# ------------------------------------------------
# Compute dx/dy from state and move_action
# ------------------------------------------------
def movement_from_state(raw_state, move_action):
    [prey,sorted_stars,sorted_predators]=raw_state
    # Pursue food
    if move_action == "PURSUIT_FOOD" and sorted_stars:
        return (prey['x']-sorted_stars[0]['x']), (prey['y']-sorted_stars[0]['y'])
    # Run from predator if exists
    if move_action == "RUN_FROM_PREDATOR" and sorted_predators:
        return (prey['x']-sorted_predators[0]['y']), (prey['y']-sorted_predators[0]['y'])
    if move_action == "RUN_TO_PREDATOR"  and sorted_predators:
        return -(prey['x']-sorted_predators[0]['y']), -(prey['x']-sorted_predators[0]['y'])
    else:
        return (random.uniform(-1, 1), random.uniform(-1, 1))

# ------------------------------------------------
# Epsilon-greedy action selection
# ------------------------------------------------
def choose_action(state_key,Q):
    if state_key in Q:
        a = int(np.argmax(Q[state_key]))
    else:
        a = int(random.randint(0,8))
        print(state_key)
    return a



matches = glob.glob(f"training_logs/q_snapshot_{EPISODE}_*.npz")
if not matches:
    raise FileNotFoundError(f"No snapshot for episode {EPISODE}")

SNAPSHOT_FILE = matches[0]  # use the first (or only) match
data = np.load(SNAPSHOT_FILE)
print(data)
Q = data
async def handler(websocket):
    async for message in websocket:
        data = json.loads(message)
        print("[PLAY] Client connected using frozen agent from episode latest")
        # otherwise process transition message
        prey_id = data.get("preyId")
        raw_state = data.get("state")
        reward = float(data.get("reward", 0))
        state_key = state_to_key(raw_state)
        terminal = data.get("terminal", False)

        # Q-update if previous step exists
        prev_state = data.get("prev_state")
        prev_action = data.get("prev_action")
        # Choose next action (shared Q)
        action_idx = choose_action(state_key,Q)
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

async def main():
    print(f"[PLAY] Loading frozen agent from: {SNAPSHOT_FILE}")
    async with websockets.serve(handler, "localhost", 9876):
        await asyncio.Future()

asyncio.run(main())
