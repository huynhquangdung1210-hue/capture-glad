import asyncio
import websockets
import json
import time
import os
from pathlib import Path
import numpy as np
from collections import deque

print("Loading RL server...")

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
# Q-Learning parameters
# ------------------------------------------------
EPSILON = 0.5
ALPHA = 0.5
GAMMA = 0.92
# Epsilon decay settings (applied per completed episode)
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.98  # multiplicative decay per episode

# Shared Q-table keyed by state_key (collective learning across all prey)
Q = {}   # { state_key -> np.array(action_values) }

# Training stats (shared across prey)
stats = {
    "total_messages": 0,
    "total_updates": 0,
    "total_reward": 0.0,
    "action_counts": [0] * ACTION_COUNT,
    "state_visits": {},
    # accumulate probability mass assigned to actions for each observed state_key
    # { state_key: [sum_prob_action0, sum_prob_action1, ...] }
    "state_action_prob_sums": {},
    # counts of times a state_key was visited (for averaging)
    "state_action_visits": {},
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
        # Prepare arrays and keys
        keys = []
        arrays = {}
        # enumerate to produce stable short keys
        for i, (k, v) in enumerate(list(Q.items())):
            keyname = f"s{i}"
            keys.append(k)
            arrays[keyname] = v

        npz_path = LOG_DIR / f"q_snapshot_{episode_idx or 'auto'}_{ts}.npz"
        # Save arrays (np.savez_compressed accepts dict of arrays)
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "timestamp": ts,
            "episode": episode_idx,
            "state_keys": keys,
            "q_size": len(keys),
        }
        with open(LOG_DIR / f"q_snapshot_{episode_idx or 'auto'}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        print(f"[RL] Saved Q snapshot: {npz_path} (entries={len(keys)})")
    except Exception as e:
        print("[RL] Failed to save Q snapshot:", e)


async def periodic_saver():
    while True:
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECS)
        save_q_snapshot()


def ensure_Q(state_key):
    """Ensure shared Q-table entry exists for a state_key."""
    if state_key not in Q:
        Q[state_key] = np.zeros(ACTION_COUNT)
    return Q[state_key]


# ------------------------------------------------
# Convert JS state → key
# ------------------------------------------------
def state_to_key(state):
    """Convert a raw state list into a compact, hashable string key.

    Rounds numeric values and converts booleans to ints to keep keys small
    and consistent between client and backend simulator.
    """
    if state is None:
        return "None"
    cleaned = []
    for i,v in enumerate(state):
        if i<4 or i==6: # discard dx/dy values and starving flag for state key
            continue
        if isinstance(v, (int, float)):
            # round floats to 3 decimals to reduce table size
            cleaned.append(round(float(v), 3))
        elif isinstance(v, bool):
            cleaned.append(int(v))
        else:
            # fallback: stringify
            try:
                cleaned.append(round(float(v), 3))
            except Exception:
                cleaned.append(str(v))
    return str(tuple(cleaned))


# ------------------------------------------------
# Epsilon-greedy action selection
# ------------------------------------------------
def choose_action(state_key):
    q = ensure_Q(state_key)
    is_random = False
    if np.random.rand() < EPSILON:
        a = int(np.random.randint(ACTION_COUNT))
        is_random = True
    else:
        a = int(np.argmax(q))

    # compute probability of the chosen action under epsilon-greedy
    if is_random:
        chosen_prob = 1.0 / ACTION_COUNT
    else:
        # greedy action gets (1 - EPSILON) + EPSILON/|A| probability
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


# ------------------------------------------------
# Q-learning update
# ------------------------------------------------
_update_debug_count = 0
_update_debug_print_every = 10_000  # prints every 1000 updates; lower to 1 for noisy output
def update_Q(prev_key, action_idx, reward, state_key, terminal=False):
    q_prev = ensure_Q(prev_key)
    if terminal:
        td_target = reward  # terminal state, no future reward
    else:
        q_next = ensure_Q(state_key)
        td_target = reward + GAMMA * np.max(q_next)
    
    td_error = td_target - q_prev[action_idx]

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



# ------------------------------------------------
# Compute dx/dy from state and move_action
# ------------------------------------------------
def movement_from_state(raw_state, move_action):
    dx_food, dy_food, dx_pred, dy_pred, energy, prednearby, starving = raw_state

    # Pursue food
    if move_action == "PURSUIT_FOOD":
        return dx_food, dy_food

    # Run from predator if exists
    if move_action == "RUN_FROM_PREDATOR" :
        return dx_pred,dy_pred
    if move_action == "RUN_TO_PREDATOR" :
        return -dx_pred,-dy_pred

# ------------------------------------------------
# WebSocket handler
# ------------------------------------------------
async def handler(websocket):
    print("Client connected")
    # Episode / games grouping: each episode equals 5 games
    GAMES_PER_EPISODE = 10
    games_completed = 0
    episodes_completed = 0
    global EPSILON
    async for message in websocket:
        data = json.loads(message)
        # allow special messages that indicate end-of-game
        msg_type = data.get("type")
        if msg_type in ("game_end", "episode_done", "gameOver"):
            games_completed += 1
            print(f"[RL] Game finished. games_completed={games_completed}/{GAMES_PER_EPISODE}")
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
    # start periodic saver task
    saver_task = asyncio.create_task(periodic_saver())
    try:
        async with websockets.serve(handler, "localhost", 8765):
            await asyncio.Future()
    finally:
        saver_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
