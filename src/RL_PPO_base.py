"""PPO (Proximal Policy Optimization) RL Server for Prey AI Training.

Server features:
- Accepts structured states: [prey_dict, stars_list, predators_list] (or flat vector).
- Preprocesses structured state into fixed-length vector (nearest N stars, M preds).
- Policy network with two heads:
    - continuous movement head (dx, dy mean + log_std param)
    - discrete food action head (logits)
- GAE-based PPO updates using combined log-prob (movement + food).
- Checkpointing and episode logging.

Compatibility:
- Listens on ws://localhost:8765 (same as previous server)
- Expects client messages with fields: `state`, `reward`, `prev_state`, `prev_action`, `preyId`, `terminal`
- Responds with: {preyId, dx, dy, action_idx, moveAction, foodAction}

Requirements:
  pip install websockets torch numpy

Run:
  python src/RL_PPO.py
"""

import asyncio
import json
import time
from pathlib import Path
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("PyTorch is required: pip install torch") from e

import websockets

# ------------------ Config ------------------
WS_HOST = "localhost"
WS_PORT = 8765

FOOD_ACTIONS = ["REPRODUCE", "ACTIVATE_SPECIAL", "IDLE_FOOD"]

# How many neighboring objects to include in state encoding
NEAREST_STARS = 3
NEAREST_PREDATORS = 3

# Flat state length: prey_x, prey_y, energy, special_flag, starvation
STATE_DIM = 5 + 2 * NEAREST_STARS + 2 * NEAREST_PREDATORS

# PPO hyperparams (tweakable)
LEARNING_RATE = 3e-4
GAMMA = 0.92
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_CLIP = 0.2
BATCH_SIZE = 64
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

TRAJECTORY_BUFFER_SIZE = 2048
UPDATE_INTERVAL = TRAJECTORY_BUFFER_SIZE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = Path("training_logs_ppo")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Networks ------------------
class Actor(nn.Module):
    """Actor with two heads: movement mean (2) and food logits (K)."""
    def __init__(self, state_dim, food_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.move_mean = nn.Linear(hidden, 2)
        # Log std as a learnable parameter (one per movement dim)
        self.move_logstd = nn.Parameter(torch.zeros(2))
        self.food_logits = nn.Linear(hidden, food_dim)

    def forward(self, state):
        h = self.shared(state)
        mean = self.move_mean(h)
        logits = self.food_logits(h)
        return mean, logits

class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)

actor = Actor(STATE_DIM, len(FOOD_ACTIONS)).to(DEVICE)
critic = Critic(STATE_DIM).to(DEVICE)
actor_opt = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

# ------------------ Buffers & stats ------------------
buffer = {
    "states": [],       # raw structured states (kept for preprocess)
    "states_flat": [],  # flat numeric arrays (numpy)
    "move_actions": [], # [dx, dy] sampled
    "food_actions": [], # int food idx sampled
    "logps": [],        # combined log-prob (movement + food)
    "values": [],
    "rewards": [],
    "dones": [],
}

stats = {"total_messages": 0, "total_updates": 0, "total_reward": 0.0, "food_counts": [0]*len(FOOD_ACTIONS)}
RECENT_RETURNS = deque(maxlen=50)
update_steps = 0
SNAPSHOT_EVERY_EPISODES=300

# ------------------ Helpers ------------------

def preprocess_structured_state(raw):
    """Map structured raw state -> fixed-length numpy array (STATE_DIM,).

    Accepts either a flat numeric list/tuple (if len==STATE_DIM) or a structured
    [prey_dict, stars_list, preds_list].
    """
    if raw is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    if isinstance(raw, (list, tuple)) and len(raw) == STATE_DIM:
        return np.array(raw, dtype=np.float32)

    prey = None
    stars = []
    preds = []
    if isinstance(raw, (list, tuple)) and len(raw) >= 1 and isinstance(raw[0], dict):
        prey = raw[0]
        if len(raw) > 1 and raw[1] is not None:
            stars = list(raw[1])
        if len(raw) > 2 and raw[2] is not None:
            preds = list(raw[2])
    elif isinstance(raw, dict):
        prey = raw

    px = float(prey.get("x", 0.0)) if prey else 0.0
    py = float(prey.get("y", 0.0)) if prey else 0.0
    energy = float(prey.get("energy", prey.get("food", 0.0))) if prey else 0.0
    special = 1.0 if (prey and prey.get("special")) else 0.0
    starvation = float(prey.get("starvationTimer", prey.get("starvationtimer", 0.0))) if prey else 0.0

    def nearest_vec(items, n):
        out=[]
        for i in range(n):
            if i < len(items):
                s = items[i]
                out.extend([float(s.get("x",0.0)-px), float(s.get("y",0.0)-py)])
            else:
                out.extend([0.0, 0.0])
        return out

    stars_v = nearest_vec(stars, NEAREST_STARS)
    preds_v = nearest_vec(preds, NEAREST_PREDATORS)
    flat = [px, py, energy, special, starvation] + stars_v + preds_v
    arr = np.array(flat, dtype=np.float32)
    if arr.shape[0] != STATE_DIM:
        if arr.shape[0] < STATE_DIM:
            arr = np.pad(arr, (0, STATE_DIM - arr.shape[0]))
        else:
            arr = arr[:STATE_DIM]
    return arr


def state_tensor(raw):
    arr = preprocess_structured_state(raw)
    return torch.from_numpy(arr).to(DEVICE)


def compute_gae_advantages(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values + [0.0], dtype=np.float32)  # bootstrap 0 for terminal
    dones = np.array(dones, dtype=np.float32)
    gae = 0.0
    advs = np.zeros_like(rewards, dtype=np.float32)
    for t in reversed(range(len(rewards))):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        advs[t] = gae
    returns = advs + values[:-1]
    if advs.std() > 1e-8:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    return advs, returns

# ------------------ PPO update ------------------
async def ppo_update():
    global update_steps
    if len(buffer["states_flat"]) < BATCH_SIZE:
        return

    states = torch.stack([torch.from_numpy(s).to(DEVICE) for s in buffer["states_flat"]])
    move_actions = torch.tensor(buffer["move_actions"], dtype=torch.float32).to(DEVICE)
    food_actions = torch.tensor(buffer["food_actions"], dtype=torch.long).to(DEVICE)
    old_logps = torch.tensor(buffer["logps"], dtype=torch.float32).to(DEVICE)
    rewards = buffer["rewards"]
    values = buffer["values"]
    dones = buffer["dones"]

    advs, returns = compute_gae_advantages(rewards, values, dones)
    advs_t = torch.from_numpy(advs).to(DEVICE)
    returns_t = torch.from_numpy(returns).to(DEVICE)

    dataset_size = states.shape[0]
    idxs = np.arange(dataset_size)

    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, dataset_size, BATCH_SIZE):
            end = min(start + BATCH_SIZE, dataset_size)
            batch = idxs[start:end]
            s = states[batch]
            a_move = move_actions[batch]
            a_food = food_actions[batch]
            oldlp = old_logps[batch]
            adv_b = advs_t[batch]
            ret_b = returns_t[batch]

            # forward
            mean, logits = actor(s)
            logstd = actor.move_logstd.unsqueeze(0).expand_as(mean)
            std = torch.exp(logstd)
            move_dist = torch.distributions.Normal(mean, std)
            logp_move = move_dist.log_prob(a_move).sum(dim=1)

            logp_food_all = F.log_softmax(logits, dim=1)
            logp_food = logp_food_all[torch.arange(len(batch)), a_food]

            logp = logp_move + logp_food

            ratio = torch.exp(logp - oldlp)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # entropy
            probs_food = F.softmax(logits, dim=1)
            entropy_food = -(probs_food * logp_food_all).sum(dim=1).mean()
            entropy_move = move_dist.entropy().sum(dim=1).mean()
            entropy = entropy_food + entropy_move

            # value loss
            vpred = critic(s)
            value_loss = F.mse_loss(vpred, ret_b)

            loss = policy_loss - ENTROPY_COEF * entropy + 0.5 * value_loss

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            actor_opt.step()
            critic_opt.step()

    update_steps += 1

    # clear buffer
    buffer["states"].clear(); buffer["states_flat"].clear()
    buffer["move_actions"].clear(); buffer["food_actions"].clear()
    buffer["logps"].clear(); buffer["values"].clear()
    buffer["rewards"].clear(); buffer["dones"].clear()

async def save_checkpoint(ep = None):
    """
    Save model checkpoint (.pt) and (when ep is provided) a JSON summary file.

    - .pt is saved with: ppo_checkpoint_{ep or 'auto'}_{ts}.pt
    - JSON is saved with: episode_{ep}_{ts}.json  (matches regex: r"episode_(\d+)_(\d+)(\.json)?$")

    JSON write is done with asyncio.to_thread to avoid blocking the event loop.

    This function attempts to use the following globals (as in typical training scripts):
      LOG_DIR (Path), actor, critic, actor_opt, critic_opt, update_steps,
      RECENT_RETURNS (iterable of recent episode returns), FOOD_ACTIONS (iterable)

    If any required training objects are missing it will raise a RuntimeError.
    """

    ts = int(time.time())

    # Resolve LOG_DIR (fallback to current directory)
    try:
        log_dir = LOG_DIR
    except NameError:
        log_dir = Path(".")

    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)

    pt_fname = log_dir / f"ppo_checkpoint_{ep or 'auto'}_{ts}.pt"

    # Build checkpoint dict (require necessary objects to exist)
    try:
        checkpoint = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "update_steps": update_steps,
            "episode": ep,
        }
    except NameError as e:
        raise RuntimeError(
            "Missing required training objects. Ensure 'actor', 'critic', 'actor_opt', "
            "'critic_opt' and 'update_steps' are defined in scope."
        ) from e

    # Save torch checkpoint off the event loop
    await asyncio.to_thread(torch.save, checkpoint, str(pt_fname))
    print(f"[PPO] Saved checkpoint: {pt_fname}")

    # When ep is provided, also create a JSON summary file
    if ep is not None:
        # Default/expected hyperparameters (use globals if they exist, otherwise defaults)
        config = {
            "LEARNING_RATE": globals().get("LEARNING_RATE", 3e-4),
            "GAMMA": globals().get("GAMMA", 0.92),
            "GAE_LAMBDA": globals().get("GAE_LAMBDA", 0.95),
            "PPO_EPOCHS": globals().get("PPO_EPOCHS", 4),
            "PPO_CLIP": globals().get("PPO_CLIP", 0.2),
            "BATCH_SIZE": globals().get("BATCH_SIZE", 64),
            "ENTROPY_COEF": globals().get("ENTROPY_COEF", 0.01),
            "MAX_GRAD_NORM": globals().get("MAX_GRAD_NORM", 0.5),
        }

        # Stats: use existing 'stats' if present, otherwise create a minimal default
        stats = globals().get(
            "stats",
            {
                "total_messages": 0,
                "total_updates": 0,
                "total_reward": 0.0,
                "food_counts": [0] * (len(globals().get("FOOD_ACTIONS", []))),
            },
        )

        # Robust recent average return computation
        recent_avg_return = None
        try:
            rr = globals().get("RECENT_RETURNS")
            if rr is None:
                recent_avg_return = 0.0
            else:
                # treat as iterable of numbers
                rr_list = list(rr)
                if len(rr_list) > 0:
                    recent_avg_return = float(sum(rr_list) / len(rr_list))
                else:
                    recent_avg_return = 0.0
        except Exception:
            recent_avg_return = None

        json_obj = {
            "timestamp": ts,
            "episode": ep,
            "checkpoint": str(pt_fname.name),
            "config": config,
            "stats": stats,
            "recent_avg_return": recent_avg_return,
        }

        json_fname = log_dir / f"episode_{ep}_{ts}.json"

        # Helper to write JSON (runs in thread via asyncio.to_thread)
        def _write_json(obj, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

        try:
            await asyncio.to_thread(_write_json, json_obj, json_fname)
            print(f"[PPO] Saved JSON summary: {json_fname}")
        except Exception as exc:
            # Don't crash training for JSON write failures, but surface a helpful message
            print(f"[PPO] Warning: failed to save JSON summary {json_fname}: {exc}")

    # Optionally return file paths (useful when calling the coroutine directly)
    return pt_fname, (json_fname if ep is not None else None)


async def load_checkpoint(path: str = None):
    """Load a checkpoint to resume training.

    If `path` is None, the function finds the most recent file in `LOG_DIR`
    whose name contains 'ppo_checkpoint' and loads it. It restores model
    and optimizer states when available and sets `update_steps`.
    Returns True if a checkpoint was loaded, False otherwise.
    """
    global update_steps
    # find file
    if path:
        candidate = Path(path)
        if not candidate.exists():
            print(f"[PPO] load_checkpoint: specified path not found: {path}")
            return False
        ckpt_path = candidate
    else:
        files = sorted(LOG_DIR.glob('ppo_checkpoint_*.pt'), key=lambda p: p.stat().st_mtime)
        if not files:
            print("[PPO] load_checkpoint: no checkpoints found")
            return False
        ckpt_path = files[-1]

    try:
        data = await asyncio.to_thread(torch.load, str(ckpt_path), map_location=DEVICE)
        if 'actor' in data:
            actor.load_state_dict(data['actor'])
        if 'critic' in data:
            critic.load_state_dict(data['critic'])
        # optimizers may not be present if checkpoint saved differently
        if 'actor_opt' in data:
            try:
                actor_opt.load_state_dict(data['actor_opt'])
            except Exception as e:
                print(f"[PPO] Warning: failed to load actor optimizer state: {e}")
        if 'critic_opt' in data:
            try:
                critic_opt.load_state_dict(data['critic_opt'])
            except Exception as e:
                print(f"[PPO] Warning: failed to load critic optimizer state: {e}")

        update_steps = int(data.get('update_steps', update_steps))
        print(f"[PPO] Loaded checkpoint: {ckpt_path} (update_steps={update_steps})")
        return True
    except Exception as e:
        print(f"[PPO] Failed to load checkpoint {ckpt_path}: {e}")
        return False

# ------------------ WebSocket handler ------------------
async def handler(ws):
    print("[PPO] client connected")
    GAMES_PER_EPISODE=1
    games_completed = 0
    episodes = 0
    async for msg in ws:
        try:
            data = json.loads(msg)
        except Exception:
            continue
        msg_type = data.get('type')
        if msg_type in ('game_end','episode_done','gameOver'):
            games_completed += 1
            if games_completed >= GAMES_PER_EPISODE:
                # append the episode total_reward to recent returns window
                try:
                    RECENT_RETURNS.append(stats.get("total_reward", 0.0))
                except Exception:
                    pass
                await ppo_update()
                episodes += 1
                games_completed = 0
                await save_checkpoint(episodes)
            continue

        prey_id = data.get('preyId')
        raw_state = data.get('state')
        reward = float(data.get('reward', 0.0))
        terminal = bool(data.get('terminal', False))

        stats['total_messages'] += 1
        stats['total_reward'] += reward

        # sample action
        s_t = state_tensor(raw_state)
        with torch.no_grad():
            mean, logits = actor(s_t.unsqueeze(0))
            mean = mean.squeeze(0)
            logits = logits.squeeze(0)
            logstd = actor.move_logstd
            std = torch.exp(logstd)

            move_dist = torch.distributions.Normal(mean, std)
            move_sample = move_dist.sample()
            logp_move = move_dist.log_prob(move_sample).sum().item()

            probs = F.softmax(logits, dim=0)
            food_dist = torch.distributions.Categorical(probs)
            food_sample = int(food_dist.sample().item())
            logp_food = float(food_dist.log_prob(torch.tensor(food_sample)).item())

            combined_logp = logp_move + logp_food
            value = float(critic(s_t.unsqueeze(0)).item())

        # store to buffer
        buffer['states'].append(raw_state)
        flat = preprocess_structured_state(raw_state)
        buffer['states_flat'].append(flat)
        buffer['move_actions'].append([float(move_sample[0].item()), float(move_sample[1].item())])
        buffer['food_actions'].append(int(food_sample))
        buffer['logps'].append(float(combined_logp))
        buffer['values'].append(float(value))
        buffer['rewards'].append(float(reward))
        buffer['dones'].append(bool(terminal))

        stats['food_counts'][food_sample] += 1
        stats['total_updates'] += 1

        # trigger update
        if len(buffer['states']) >= UPDATE_INTERVAL:
            await ppo_update()

        # decode discrete action to moveAction and foodAction for compatibility
        move_idx = (food_sample) // len(FOOD_ACTIONS)  # not meaningful now but kept
        # choose move behavior string
        move_action_str = 'CONTINUOUS'
        food_action_str = FOOD_ACTIONS[food_sample]

        dx = float(move_sample[0].item())
        dy = float(move_sample[1].item())

        resp = {
            'preyId': prey_id,
            'dx': dx,
            'dy': dy,
            'action_idx': int(food_sample),
            'moveAction': move_action_str,
            'foodAction': food_action_str
        }
        await ws.send(json.dumps(resp))

# ------------------ Main ------------------
async def main():
    print(f"[PPO] Running on ws://{WS_HOST}:{WS_PORT} device={DEVICE}")
    # attempt to load latest checkpoint (non-blocking) so training continues after restart
    loaded = await load_checkpoint()
    if not loaded:
        print("[PPO] No checkpoint loaded; starting from scratch")
    async with websockets.serve(handler, WS_HOST, WS_PORT):
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
