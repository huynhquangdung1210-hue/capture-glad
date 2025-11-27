import asyncio
import websockets
import json
import numpy as np
import os
import time
from pathlib import Path

print("Loading RL server...")

# ------------------------------------------------
# ACTION SPACE
# ------------------------------------------------
# Movement types (these do NOT contain dx/dy)
MOVE_ACTIONS = [
    "PURSUIT_FOOD",
    "RUN_FROM_PREDATOR",
    "TOWARD_PREDATOR",
    "WANDER"
]

FOOD_ACTIONS = ["REPRODUCE", "ACTIVATE_SPECIAL", "IDLE_FOOD"]

ACTION_COUNT = len(MOVE_ACTIONS) * len(FOOD_ACTIONS)


def decode_action(idx):
    move_idx = idx // len(FOOD_ACTIONS)
    food_idx = idx % len(FOOD_ACTIONS)
    return MOVE_ACTIONS[move_idx], FOOD_ACTIONS[food_idx]
import math
import random

def normalize(x, y):
    m = math.hypot(x, y)
    return (0.0, 0.0) if m == 0 else (x/m, y/m)

def smooth_random_direction(dx_food, dy_food, dx_pred, dy_pred):
    # Combine into deterministic steering direction
    bx = dx_food * 0.4 + dx_pred * 1.0
    by = dy_food * 0.4 + dy_pred * 1.0

    # Normalize the base direction
    bx, by = normalize(bx, by)

    # ------------------------------------------
    # Smooth wandering:
    # Add noise PERPENDICULAR to the base vector
    # ------------------------------------------
    # Perpendicular vector (rotate by 90 degrees)
    px = -by
    py = bx

    # Small signed amount: [-curviness, curviness]
    CURVE = 0.01       # smaller = smoother, slower turn
    k = (random.random() * 2 - 1) * CURVE

    # Add small perpendicular bend
    bx2 = bx + px * k
    by2 = by + py * k

    # Normalize final
    return normalize(bx2, by2)


# ------------------------------------------------
# Q-Learning parameters
# ------------------------------------------------
EPSILON = 0.08
ALPHA = 0.25
GAMMA = 0.92

class RLModel:
    """Encapsulates the shared Q-table and training utilities.

    This class owns the Q dictionary and training stats, and provides methods
    compatible with the earlier `select_action` / `store_transition` style.
    """
    def __init__(self, snapshot_dir=Path("training_logs"), snapshot_interval=60, snapshot_updates=1000):
        self.Q = {}
        self.SNAPSHOT_DIR = Path(snapshot_dir)
        self.SNAPSHOT_INTERVAL_SECONDS = snapshot_interval
        self.SNAPSHOT_UPDATES = snapshot_updates
        self.stats = {
            "total_messages": 0,
            "total_updates": 0,
            "total_reward": 0.0,
            "action_counts": [0] * ACTION_COUNT,
            "state_visits": {},
        }

    def ensure_snapshot_dir(self):
        try:
            self.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def ensure_Q(self, state_key):
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(ACTION_COUNT)
            self.stats["state_visits"][state_key] = 0
        return self.Q[state_key]

    def choose_action(self, cooldown, state_key):
        q = self.ensure_Q(state_key)
        if cooldown>0:
            return int(np.argmax(q[9:]))
        if np.random.rand() < EPSILON:
            return int(np.random.randint(ACTION_COUNT))
        return int(np.argmax(q))

    def update_Q(self, prev_key, action_idx, reward, state_key):
        q_prev = self.ensure_Q(prev_key)
        q_next = self.ensure_Q(state_key)
        td_target = reward + GAMMA * np.max(q_next)
        td_error = td_target - q_prev[action_idx]
        q_prev[action_idx] += ALPHA * td_error
        # update training stats
        try:
            self.stats["total_updates"] += 1
            self.stats["total_reward"] += float(reward)
            self.stats["state_visits"][prev_key] = self.stats["state_visits"].get(prev_key, 0) + 1
        except Exception:
            pass

    def select_action(self, prey_id, state):
        """Select action given the raw state (keeps handler interface)."""
        state_key = state_to_key(state)
        _,_,_,_,cooldown,_,_,_ = state
        action_idx = self.choose_action(cooldown,state_key)
        move_action, food_action = decode_action(action_idx)
        dx, dy,cooldown = movement_from_state(state, move_action)
        # print("This is after movement from state:",cooldown)
        return {"dx": dx, "dy": dy, "cooldown":cooldown,"action_idx": action_idx, "moveAction": move_action, "foodAction": food_action}

    def store_transition(self, prey_id, prev_state, prev_action_idx, reward, state, done=False):
        """Record/update Q from a transition."""
        if prev_state is not None and prev_action_idx is not None:
            prev_key = state_to_key(prev_state)
            state_key = state_to_key(state)
            self.update_Q(prev_key, prev_action_idx, reward, state_key)

    def save_snapshot(self):
        self.ensure_snapshot_dir()
        ts = int(time.time())
        base = self.SNAPSHOT_DIR / f"snapshot_{ts}_u{self.stats['total_updates']}"
        try:
            # Make a stable snapshot of Q items so concurrent updates don't
            # mutate the dict during iteration (causes "dictionary changed size"
            # errors). We take a shallow copy of items and stats.
            q_items = list(self.Q.items())
            q_json = {k: v.tolist() for k, v in q_items}
            stats_copy = {
                "total_messages": self.stats.get("total_messages", 0),
                "total_updates": self.stats.get("total_updates", 0),
                "total_reward": self.stats.get("total_reward", 0.0),
                "action_counts": list(self.stats.get("action_counts", [])),
                "state_visits": dict(self.stats.get("state_visits", {})),
            }
            with open(str(base) + ".json", "w", encoding="utf-8") as f:
                json.dump({"q": q_json, "stats": stats_copy}, f)

            # Save Q arrays into a compressed archive using stable iteration order
            npz_dict = {f"s{i}": v for i, (k, v) in enumerate(q_items)}
            np.savez_compressed(str(base) + ".npz", **npz_dict)
        except Exception as e:
            print("Failed to save snapshot:", e)

    async def periodic_saver(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                await asyncio.sleep(self.SNAPSHOT_INTERVAL_SECONDS)
                await loop.run_in_executor(None, self.save_snapshot)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print("Error in periodic_saver:", e)


rl_model = RLModel()


# ------------------------------------------------
# Convert JS state → key
# ------------------------------------------------
def state_to_key(state):
    # Example state format:
    # [dx_food, dy_food, pred_dx,pred_dy,nearpred, starving_flag]
    cleaned = []
    for i, v in enumerate(state):
        if i<4:
            continue 
        if isinstance(v, (int, float)):
            cleaned.append(round(float(v), 3))
        else:
            cleaned.append(v)
    return str(tuple(cleaned))


# ------------------------------------------------
# Epsilon-greedy action selection
# ------------------------------------------------
def choose_action(prey_id, state_key):
    # delegate to RLModel instance (shared Q)
    return rl_model.choose_action(state_key)


# ------------------------------------------------
# Q-learning update
# ------------------------------------------------
def update_Q(prey_id, prev_key, action_idx, reward, state_key):
    # delegate to RLModel (shared Q table)
    return rl_model.update_Q(prev_key, action_idx, reward, state_key)



# ------------------------------------------------
# Compute dx/dy from state and move_action
# ------------------------------------------------
def movement_from_state(raw_state, move_action):
    dx_food, dy_food, dx_pred, dy_pred, cooldown, energy, prednearby, starving = raw_state
        # Direct movement toward/away without wandering
    if move_action == "PURSUIT_FOOD":
        return dx_food, dy_food,cooldown
    if move_action == "RUN_FROM_PREDATOR":
        return dx_pred, dy_pred,cooldown
    if move_action == "TOWARD_PREDATOR":
        return -dx_pred, -dy_pred,cooldown
    if move_action == "WANDER":
        dx_wander, dy_wander = smooth_random_direction(0, 0, dx_pred, dy_pred)
        cooldown=cooldown-1 if cooldown>0 else 5
        return dx_wander, dy_wander,cooldown


# ------------------------------------------------
# WebSocket handler
# ------------------------------------------------
# ===============================
# WebSocket handler for prey
# ===============================
async def handler(websocket):
    print("[RL Server] Client connected")

    NUM_EPISODES = 1000
    MAX_EPISODE_TIME = 30  # seconds per episode

    for episode in range(NUM_EPISODES):
        print(f"[RL] Starting episode {episode}")
        episode_start = time.time()
        done = False

        # Loop over ticks for this episode — use explicit recv so we can catch
        # connection-close exceptions and exit cleanly instead of letting
        # `async for` hide the lifecycle.
        while not done:
            try:
                message = await websocket.recv()
            except websockets.exceptions.ConnectionClosed as e:
                print("[RL] Connection closed by client:", e)
                return

            try:
                data = json.loads(message)
            except Exception:
                # malformed message, skip
                continue

            prey_id = data.get("preyId")
            raw_state = data.get("state")
            reward = float(data.get("reward", 0))
            prev_state = data.get("prev_state")
            prev_action = data.get("prev_action")

            # Track messages
            try:
                rl_model.stats["total_messages"] += 1
            except Exception:
                pass

            # Store transition
            try:
                rl_model.store_transition(prey_id, prev_state, prev_action, reward, raw_state, done=False)
            except Exception:
                pass

            # Select next action
            try:
                action = rl_model.select_action(prey_id, raw_state)
            except Exception as e:
                print("[RL] select_action failed:", e)
                return

            action_idx = int(action.get("action_idx", 0))
            dx = action.get("dx", 0)
            dy = action.get("dy", 0)
            cooldown = action.get("cooldown", 0)
            # print("This is before sending to js:", cooldown)
            move_action, food_action = decode_action(action_idx)

            # Track action counts
            try:
                rl_model.stats["action_counts"][action_idx] += 1
            except Exception:
                pass

            # Send action back to JS, but guard against client disconnects
            try:
                await websocket.send(json.dumps({
                    "preyId": prey_id,
                    "dx": dx,
                    "dy": dy,
                    "cooldown": cooldown,
                    "action_idx": action_idx,
                    "moveAction": move_action,
                    "foodAction": food_action
                }))
            except websockets.exceptions.ConnectionClosed as e:
                print("[RL] send failed, connection closed:", e)
                return

            # Save snapshot periodically (run in executor to avoid blocking)
            try:
                if rl_model.SNAPSHOT_UPDATES > 0 and rl_model.stats["total_updates"] % rl_model.SNAPSHOT_UPDATES == 0 and rl_model.stats["total_updates"] > 0:
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, rl_model.save_snapshot)
            except Exception:
                pass

            # Check episode time
            now = time.time()
            if now - episode_start > MAX_EPISODE_TIME:
                done = True
                print(f"[RL] Episode {episode} finished (time limit)")
                # Optionally notify JS (ignore failure if client closed)
                try:
                    await websocket.send(json.dumps({"type": "episode_done"}))
                except Exception:
                    pass
                break  # start next episode

# ===============================
# Run WebSocket server
# ===============================
async def main():
    print("[RL Server] Starting on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())