"""Tabular Q-Learning RL Server for Prey AI Training.

This server implements a classic Q-learning algorithm using a table-based approach.
It connects via WebSocket to receive state transitions from the game client and
sends back discrete actions that prey agents should take.

Key Features:
- Epsilon-greedy exploration with decay
- Temporal difference (TD) learning with adjustable learning rate
- State discretization to reduce state space size
- Shared Q-table across all prey for collective learning
- Periodic checkpointing and episode logging
- Action space: combination of movement strategies and food actions

Requirements:
  pip install websockets numpy

Usage:
  python src/RL_NAIVE.py
"""

import asyncio
import websockets
import json
import time
import os
from pathlib import Path
import numpy as np
from collections import deque
import math
import random

print("Loading RL server...")

# ================================================
# ACTION SPACE CONFIGURATION
# ================================================
# The action space is a Cartesian product of movement and food actions.
# Total actions = len(MOVE_ACTIONS) × len(FOOD_ACTIONS)

# Movement strategies (direction is computed from state, not stored in action)
MOVE_ACTIONS = [
    "PURSUIT_FOOD",      # Move towards nearest star
    "RUN_FROM_PREDATOR", # Move away from nearest predator
    "RUN_TO_PREDATOR",   # Aggressive: move towards predator (risky)
]

# Food/ability actions that prey can execute
FOOD_ACTIONS = [
    "REPRODUCE",         # Spawn offspring (costs energy)
    "ACTIVATE_SPECIAL",  # Become invincible and kill nearby predators
    "IDLE_FOOD"          # No special action
]

# Total number of discrete actions available
ACTION_COUNT = len(MOVE_ACTIONS) * len(FOOD_ACTIONS)


def decode_action(idx):
    """Convert flat action index to (movement, food) action tuple.
    
    Args:
        idx: Integer action index in [0, ACTION_COUNT)
    
    Returns:
        Tuple of (move_action_str, food_action_str)
    """
    move_idx = idx // len(FOOD_ACTIONS)
    food_idx = idx % len(FOOD_ACTIONS)
    return MOVE_ACTIONS[move_idx], FOOD_ACTIONS[food_idx]


# ================================================
# Q-LEARNING HYPERPARAMETERS
# ================================================

# Episode tracking
episodes_completed = 0

# EPSILON: Exploration rate (probability of random action)
# Higher values = more exploration, lower values = more exploitation
EPSILON = 0.5

# ALPHA: Learning rate (how much to update Q-values)
# Higher values = faster learning but less stable
ALPHA = 0.5

# GAMMA: Discount factor (importance of future rewards)
# Higher values = more long-term thinking
GAMMA = 0.92

# Epsilon decay: gradually reduce exploration over time
EPSILON_MIN = 0.01      # Don't explore less than this
EPSILON_DECAY = 0.98    # Multiply epsilon by this after each episode

# ================================================
# Q-TABLE AND STATISTICS
# ================================================

# Shared Q-table: all prey contribute to and learn from the same table
# This enables collective learning - knowledge gained by one prey benefits all
# Key: discretized state string, Value: numpy array of Q-values for each action
Q = {}   # { state_key: np.array([Q(s,a0), Q(s,a1), ..., Q(s,aN)]) }

# Training statistics for monitoring and analysis
stats = {
    "total_messages": 0,           # Total state messages received
    "total_updates": 0,            # Total Q-value updates performed
    "total_reward": 0.0,           # Cumulative reward across all prey
    "action_counts": [0] * ACTION_COUNT,  # How often each action was chosen
    "state_visits": {},            # Visit counts per state
    
    # Track action selection probabilities for analysis
    # Helps understand if agent is converging to deterministic policy
    "state_action_prob_sums": {},  # Cumulative probability per (state, action)
    "state_action_visits": {},     # Visit counts for averaging probabilities
}

# Track recent episode returns (for averaging)
RECENT_EPISODE_RETURNS = deque(maxlen=50)

# folder for episode logs
LOG_DIR = Path("training_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Snapshot settings
SNAPSHOT_EVERY_EPISODES = 1  # save snapshot every N episodes
SNAPSHOT_INTERVAL_SECS = 300  # also save snapshot every 5 minutes


def save_q_snapshot(episode_idx=None):
    """Save Q (numpy arrays) and a small metadata JSON mapping.

    Produces two files:
      - training_logs/q_snapshot_<episode>_<ts>.rewa
      - training_logs/q_snapshot_<episode>_<ts>.json
    """
    try:
        ts = int(time.time())
        # # Prepare arrays and keys
        # keys = []
        # arrays = {}
        # # enumerate to produce stable short keys
        # for i, (k, v) in enumerate(list(Q.items())):
        #     keyname = str(Q[i])
        #     keys.append(k)
        #     arrays[keyname] = v

        npz_path = LOG_DIR / f"q_snapshot_{episode_idx or 'auto'}_{ts}.npz"
        # Save arrays (np.savez_compressed accepts dict of arrays)
        np.savez_compressed(npz_path, **Q)

        meta = {
            "timestamp": ts,
            "episode": episode_idx,
            "state_keys": list(Q.keys()),
            "q_size": len(list(Q)),
            "epsilon": float(EPSILON),


        }
        with open(LOG_DIR / f"q_snapshot_{episode_idx or 'auto'}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print(f"[RL] Saved Q snapshot: {npz_path} (entries={len(list(Q.keys()))})")
    except Exception as e:
        print("[RL] Failed to save Q snapshot:", e)


async def load_q_checkpoint(path: str = None):
    """Load the most recent Q snapshot (npz) into the global Q dict.

    If `path` is provided, it will be used directly. Otherwise the function
    searches `LOG_DIR` for files matching `q_snapshot_*.npz` and loads the
    newest one. Returns True if a checkpoint was loaded, False otherwise.
    """
    # find file
    global episodes_completed, EPSILON
    if path:
        candidate = Path(path)
        if not candidate.exists():
            print(f"[RL] load_q_checkpoint: specified path not found: {path}")
            return False
        npz_path = candidate
    else:
        files = sorted(LOG_DIR.glob('q_snapshot_*.npz'), key=lambda p: p.stat().st_mtime)
        if not files:
            print("[RL] load_q_checkpoint: no checkpoints found")
            return False
        npz_path = files[-1]

    try:
        # load in thread to avoid blocking event loop
        data = await asyncio.to_thread(np.load, str(npz_path), allow_pickle=True)

        # clear existing Q and populate from file
        Q.clear()

        for name in data.files:
            try:
                arr = data[name]
                # convert to numpy array (ensure numeric dtype)
                Q[name] = np.array(arr)
            except Exception as e:
                print(f"[RL] Warning: failed to load key {name} from {npz_path}: {e}")

        try:
            episodic_files = sorted(LOG_DIR.glob('episode_*.json'), key=lambda p: p.stat().st_mtime)
            meta_text = await asyncio.to_thread(episodic_files[-1].read_text)
            meta = json.loads(meta_text)
            metainfos=meta.get('meta')
            EPSILON=metainfos['epsilon']
            episodes_completed= meta.get('episode')
            print(f"[RL] Loaded snapshot meta: episode={episodes_completed} entries={EPSILON}")
        except Exception:
            pass

        print(f"[RL] Loaded Q snapshot: {npz_path} (entries={len(Q)})")
        return True
    except Exception as e:
        print(f"[RL] Failed to load Q snapshot {npz_path}: {e}")
        return False


async def periodic_saver():
    while True:
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECS)
        save_q_snapshot()


def ensure_Q(state_key):
    """Ensure shared Q-table entry exists for a state_key."""
    if state_key not in Q:
        Q[state_key] = np.zeros(ACTION_COUNT)
    return Q[state_key]


# ================================================
# STATE DISCRETIZATION
# ================================================

# Distance threshold for "predator nearby" binary feature
predatornearbycutoff = 100  # pixels

def state_to_key(state):
    """Convert continuous game state to discrete string key for Q-table lookup.
    
    State discretization is crucial for tabular Q-learning. The full game state
    (positions, velocities, etc.) is too large to store in a table. We extract
    only the most relevant features and discretize them.
    
    Features extracted:
    - Prey energy level (discrete integer)
    - Predator nearby (boolean: is nearest predator within cutoff distance?)
    - Distance to arena center (rounded to nearest 100 pixels)
    
    Args:
        state: List of [prey_dict, sorted_stars_list, sorted_predators_list]
    
    Returns:
        String representation of discretized state (hashable for dict key)
    """
    [prey, sorted_stars, sorted_predators] = state
    
    if state is None:
        return "None"
    
    # Feature 1: Distance to nearest predator (binary: near vs far)
    dist_nearestpred = math.sqrt(
        (prey['x'] - sorted_predators[0]['x'])**2 + 
        (prey['y'] - sorted_predators[0]['y'])**2
    ) if sorted_predators else 300
    prednearby = int(dist_nearestpred < predatornearbycutoff)
    
    # Feature 2: Current energy level
    preyenergy = prey['energy']
    
    # Feature 3: Distance from arena center (discretized to nearest 100)
    # This helps prey learn to avoid arena edges
    dist_prey_center = round(float(math.sqrt(prey["x"]**2 + prey["y"]**2)), -2)
    
    # Combine features into tuple and convert to string for dict key
    cleaned = [preyenergy, prednearby, dist_prey_center]
    return str(tuple(cleaned))


# ================================================
# ACTION SELECTION (EPSILON-GREEDY)
# ================================================

def choose_action(state_key):
    """Select action using epsilon-greedy policy.
    
    Epsilon-greedy balances exploration (trying random actions) with
    exploitation (using best known action). With probability EPSILON,
    choose random action; otherwise choose action with highest Q-value.
    
    Args:
        state_key: Discretized state string
    
    Returns:
        Integer action index in [0, ACTION_COUNT)
    """
    # Get Q-values for this state (creates zeros if new state)
    q = ensure_Q(state_key)
    
    is_random = False
    
    # Exploration: random action with probability EPSILON
    if np.random.rand() < EPSILON:
        a = int(np.random.randint(ACTION_COUNT))
        is_random = True
    # Exploitation: choose best action (highest Q-value)
    else:
        a = int(np.argmax(q))

    # Compute probability of chosen action for statistics
    # This helps track whether policy is becoming more deterministic
    if is_random:
        # Uniform probability across all actions
        chosen_prob = 1.0 / ACTION_COUNT
    else:
        # Greedy action gets most probability mass
        # P(greedy) = (1 - ε) + ε/|A|  (greedy + random component)
        chosen_prob = (1.0 - EPSILON) + (EPSILON / ACTION_COUNT)

    # record chosen action and accumulate probability
    try:
        stats["action_counts"][a] += 1
        # initialize per-state arrays lazily
        if state_key not in stats["state_action_prob_sums"]:
            stats["state_action_prob_sums"][state_key] = [0.0] * ACTION_COUNT
            stats["state_action_visits"][state_key] = 0
        stats["state_action_prob_sums"][state_key][a] += float(chosen_prob)
        stats["state_action_visits"][state_key] += 1
    except Exception:
        pass

    return a


# ================================================
# Q-LEARNING UPDATE (TEMPORAL DIFFERENCE LEARNING)
# ================================================

# Debug counters for monitoring update frequency
_update_debug_count = 0
_update_debug_print_every = 10_000  # Print stats every N updates

def update_Q(prev_key, action_idx, reward, state_key, terminal=False):
    """Update Q-value using TD(0) learning rule.
    
    Q-learning update formula:
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    
    Where:
    - s: previous state (prev_key)
    - a: action taken (action_idx)
    - r: reward received
    - s': new state (state_key)
    - α (ALPHA): learning rate
    - γ (GAMMA): discount factor
    
    For terminal states, there's no future reward, so we just use r.
    
    Args:
        prev_key: Previous state key
        action_idx: Action that was taken
        reward: Reward received from transition
        state_key: New state reached
        terminal: Whether new state is terminal (episode end)
    """
    # Get Q-values for previous state
    q_prev = ensure_Q(prev_key)
    
    # Compute TD target (what Q-value should be)
    if terminal:
        # Terminal state: no future rewards possible
        td_target = reward
    else:
        # Non-terminal: add discounted value of best future action
        q_next = ensure_Q(state_key)
        td_target = reward + GAMMA * np.max(q_next)
    
    # Compute TD error: difference between target and current estimate
    td_error = td_target - q_prev[action_idx]

    # Update Q-value: move towards target by learning rate
    q_prev[action_idx] += ALPHA * td_error
    # update training stats
    # global _update_debug_count
    # _update_debug_count += 1
    # if _update_debug_count % _update_debug_print_every == 0:
    #     print("Total reward",stats["total_reward"])
    try:
        stats["total_updates"] += 1
        stats["total_reward"] += float(reward)
        stats["state_visits"][prev_key] = stats["state_visits"].get(prev_key, 0) + 1
    except Exception:
        pass


# def update_Q(prev_key, action_idx, reward, state_key, terminal=False):
#     global _update_debug_count
#     q_prev = ensure_Q(prev_key)
#     q_prev_before = q_prev[action_idx].copy()

#     if terminal:
#         td_target = reward
#     else:
#         q_next = ensure_Q(state_key)
#         td_target = reward + GAMMA * np.max(q_next)

#     td_error = td_target - q_prev[action_idx]
#     q_prev[action_idx] += ALPHA * td_error

#     _td_error_bucket.append(td_error)
#     _update_debug_count += 1

#     # if _update_debug_count % _update_debug_print_every == 0:
#     #     print(f"update #{_update_debug_count} sample:")
#     #     print(" prev_key:", prev_key)
#     #     print(" action_idx:", action_idx)
#     #     print(" reward:", reward)
#     #     print(" next_key:", state_key)
#     #     print(" q_prev_before:", q_prev_before)
#     #     print(" td_target:", td_target, "td_error:", td_error, "update:", ALPHA*td_error)
#     #     print(" q_prev_after:", q_prev[action_idx])
#     #     # quick stats
#     #     arr = np.array(_td_error_bucket[-1000:])
#     #     print(" recent td_error mean/std/min/max (last 1000):", arr.mean(), arr.std(), arr.min(), arr.max())



# ================================================
# MOVEMENT COMPUTATION
# ================================================

def movement_from_state(raw_state, move_action):
    """Convert high-level movement action to concrete (dx, dy) direction.
    
    The RL agent selects a movement strategy (e.g., "pursue food"),
    and this function translates it to actual movement direction based
    on current game state.
    
    Args:
        raw_state: [prey_dict, sorted_stars, sorted_predators]
        move_action: One of MOVE_ACTIONS strings
    
    Returns:
        Tuple (dx, dy) representing movement direction
    """
    [prey, sorted_stars, sorted_predators] = raw_state
    
    # Strategy 1: Move towards nearest star (food)
    if move_action == "PURSUIT_FOOD":
        return (prey['x'] - sorted_stars[0]['x']), (prey['y'] - sorted_stars[0]['y'])
    
    # Strategy 2: Run away from nearest predator
    if move_action == "RUN_FROM_PREDATOR":
        return (prey['x'] - sorted_predators[0]['y']), (prey['y'] - sorted_predators[0]['y'])
    
    # Strategy 3: Move towards predator (risky, but might use special ability)
    if move_action == "RUN_TO_PREDATOR":
        return -(prey['x'] - sorted_predators[0]['y']), -(prey['x'] - sorted_predators[0]['y'])
    
    # Fallback: random movement
    else:
        return (random.uniform(-1, 1), random.uniform(-1, 1))

# ------------------------------------------------
# WebSocket handler
# ------------------------------------------------
async def handler(websocket):
    print("Client connected")
    # Episode / games grouping: each episode equals 5 games
    GAMES_PER_EPISODE = 10
    games_completed = 0
    global episodes_completed, EPSILON
    async for message in websocket:
        data = json.loads(message)
        # allow special messages that indicate end-of-game
        msg_type = data.get("type")
        if msg_type in ("game_end", "episode_done", "gameOver"):
            games_completed += 1
            print(f"[RL] Game finished. games_completed={games_completed}/{GAMES_PER_EPISODE}. Episode: {episodes_completed}. EPSILON: {EPSILON}")
            # When we complete an episode (GAMES_PER_EPISODE games), print/save training log
            if games_completed >= GAMES_PER_EPISODE:
                episodes_completed += 1
                ts = int(time.time())
                log = {
                    "timestamp": ts,
                    "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                    "episode": episodes_completed,
                    "games_in_episode": games_completed,
                    "stats": {
                        "total_messages": stats.get("total_messages", 0),
                        "total_updates": stats.get("total_updates", 0),
                        "total_reward": stats.get("total_reward", 0.0),
                        "action_counts": stats.get("action_counts", []),
                        "state_visits": stats.get("state_visits", {}),
                        "state_action_prob_sums": stats.get("state_action_prob_sums", {}),
                        "state_action_visits": stats.get("state_action_visits", {}),
                    }
                }
                # print JSON to console
                try:
                    # Add Q summary and average return before printing/saving
                    # compute average return over recent episodes
                    recent_avg_return = None
                    try:
                        if len(RECENT_EPISODE_RETURNS) > 0:
                            recent_avg_return = float(sum(RECENT_EPISODE_RETURNS) / len(RECENT_EPISODE_RETURNS))
                        else:
                            recent_avg_return = 0.0
                    except Exception:
                        recent_avg_return = None

                    q_list = {}
                    for k, arr in Q.items():
                        try:
                            values = [float(x) for x in arr] # convert numpy array → Python list
                            q_list[k] = values  # store under state_key
                        except Exception:
                            continue

                    # extend the log with diagnostics
                    log["meta"] = {
                        "recent_avg_return": recent_avg_return,
                        "epsilon": float(EPSILON),
                        "q_summary": q_list,
                    }
                    print("[RL][EPISODE_LOG] " + json.dumps(log))
                except Exception:
                    # print("[RL][EPISODE_LOG] (failed to json-dump)")
                    # print("[RL][EPISODE_LOG] JSON ERROR:", Exception)
                    # print("Offending log object:")
                    try:
                        import pprint
                        pprint.pprint(log)
                    except:
                        print("Cannot pprint log")
                # also save to file for later visualization
                try:
                    fname = LOG_DIR / f"episode_{episodes_completed}_{ts}.json"
                    await asyncio.to_thread(json.dump, log, open(fname, "w", encoding="utf-8"))

                except Exception as e:
                    print("[RL] Failed to write episode log:", e)

                # append the episode total_reward to recent returns window
                try:
                    RECENT_EPISODE_RETURNS.append(stats.get("total_reward", 0.0))
                except Exception:
                    pass

                # save Q snapshot periodically
                try:
                    if SNAPSHOT_EVERY_EPISODES and episodes_completed % SNAPSHOT_EVERY_EPISODES == 0:
                        save_q_snapshot(episodes_completed)
                except Exception as e:
                    print("[RL] Failed to save snapshot after episode:", e)

                # decay epsilon each episode (multiplicative decay)
                try:
                    # global EPSILON
                    EPSILON = max(EPSILON_MIN, float(EPSILON) * EPSILON_DECAY)
                except Exception:
                    pass

                # reset per-episode aggregated stats (keep Q intact)
                stats["total_messages"] = 0
                stats["total_updates"] = 0
                stats["total_reward"] = 0.0
                stats["action_counts"] = [0] * ACTION_COUNT
                stats["state_visits"] = {}
                games_completed = 0
            # continue to next message
            continue
        # otherwise process transition message
        prey_id = data.get("preyId")
        raw_state = data.get("state")
        reward = float(data.get("reward", 0))
        state_key = state_to_key(raw_state)
        terminal = data.get("terminal", False)

        # Q-update if previous step exists
        prev_state = data.get("prev_state")
        prev_action = data.get("prev_action")

        # track total messages
        try:
            stats["total_messages"] += 1
        except Exception:
            pass

        if prev_state is not None and prev_action is not None:
            try:
                prev_key = state_to_key(prev_state)
                prev_action_idx = int(prev_action)
                update_Q(prev_key, prev_action_idx, reward, state_key, terminal)
                await asyncio.sleep(0)  # yield control to keep WebSocket alive

            except Exception:
                # ignore malformed prev_action
                pass

        # Choose next action (shared Q)
        action_idx = choose_action(state_key)
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
    # attempt to load latest Q snapshot so training can continue after restart
    loaded = await load_q_checkpoint()
    if not loaded:
        print("[RL] No Q snapshot loaded; starting from scratch")

    # start periodic saver task
    saver_task = asyncio.create_task(periodic_saver())
    try:
        async with websockets.serve(handler, "localhost", 8765, ping_timeout=300,ping_interval=60):
            await asyncio.Future()
    finally:
        saver_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
