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
LEARNING_RATE = 3e-4  # Moderate: Policy is updating now, don't overshoot
CRITIC_LEARNING_RATE = 1e-3  # Higher than actor to help value learning
GAMMA = 0.98  # Increased to value long-term rewards more
GAE_LAMBDA = 0.95  # Standard value for stable learning
PPO_EPOCHS = 4  # Standard for good convergence
PPO_CLIP = 0.2
BATCH_SIZE = 64
ENTROPY_COEF = 0.01  # Moderate exploration
VALUE_LOSS_COEF = 1.0  # Balanced with policy loss
MAX_GRAD_NORM = 0.5  # Restore clipping to prevent instability
TARGET_KL = 0.015  # Standard PPO target

# Optional: per-head entropy weighting (keeps discrete head exploring)
ENTROPY_COEF_MOVE = 0.005
ENTROPY_COEF_FOOD = 0.02

TRAJECTORY_BUFFER_SIZE = 2048
UPDATE_INTERVAL = TRAJECTORY_BUFFER_SIZE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = Path("training_logs_ppo")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ Networks ------------------
class Actor(nn.Module):
    """Actor with two heads: movement mean (2) and food logits (K)."""
    def __init__(self, state_dim, food_dim, hidden=128, init_log_std=-0.5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.move_mean = nn.Linear(hidden, 2)
        # Log std as a learnable parameter (init to ~0.6 std for exploration)
        self.move_logstd = nn.Parameter(torch.ones(2) * init_log_std)
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
critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)

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
LAST_UPDATE_METRICS = {}

# Reward normalization stats
reward_running_mean = 0.0
reward_running_std = 1.0
reward_count = 0
REWARD_NORM_CLIP = 10.0  # Clip normalized rewards
REWARD_NORM_WARMUP = 100  # Don't normalize until we have enough samples
REWARD_SCALE = 0.1  # INCREASED: Less aggressive scaling
USE_REWARD_NORMALIZATION = False  # DISABLED: Eliminate normalization issues

# Optional: small epsilon-greedy for food action sampling
USE_FOOD_EPSILON = True
FOOD_EPSILON = 0.02

# ------------------ Helpers ------------------

def normalize_reward(reward):
    """Normalize rewards using running statistics to stabilize training.
    
    CRITICAL: Only normalizes after warmup period to avoid learning on noise.
    Also applies a fixed scale factor to keep rewards in reasonable range.
    
    Can be disabled by setting USE_REWARD_NORMALIZATION = False for debugging.
    """
    global reward_running_mean, reward_running_std, reward_count
    
    if not USE_REWARD_NORMALIZATION:
        # Just apply scaling, no normalization
        return reward * REWARD_SCALE
    
    reward_count += 1
    
    # Apply fixed scaling first (prevents raw reward explosion)
    scaled_reward = reward * REWARD_SCALE
    
    # Update running statistics using Welford's algorithm
    delta = scaled_reward - reward_running_mean
    reward_running_mean += delta / reward_count
    delta2 = scaled_reward - reward_running_mean
    reward_running_var = max(1e-8, reward_running_std**2 + (delta * delta2 - reward_running_std**2) / reward_count)
    reward_running_std = np.sqrt(reward_running_var)
    
    # During warmup, just use scaled reward
    if reward_count < REWARD_NORM_WARMUP:
        normalized = scaled_reward
    else:
        # Normalize with a minimum std to prevent explosion
        # Use smaller min_std to allow normalization to work
        std_for_norm = max(reward_running_std, 0.01)
        normalized = (scaled_reward - reward_running_mean) / std_for_norm
    
    # Log periodically
    if reward_count % 1000 == 0:
        mode = "scaled" if reward_count < REWARD_NORM_WARMUP else "normalized"
        print(f"[Reward Norm] count={reward_count}, mode={mode}, mean={reward_running_mean:.4f}, std={reward_running_std:.4f}, "
              f"raw={reward:.4f}, scaled={scaled_reward:.4f}, output={normalized:.4f}")
    
    return np.clip(normalized, -REWARD_NORM_CLIP, REWARD_NORM_CLIP)

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
    
    # AGGRESSIVE: Less normalization to preserve gradient signal
    if len(advs) > 1:
        adv_mean = advs.mean()
        adv_std = advs.std()
        # Only normalize if std is substantial
        if adv_std > 0.1:
            advs = (advs - adv_mean) / (adv_std + 1e-8)
        else:
            # Keep raw advantages if std is too small
            print(f"[GAE] WARNING: Advantage std={adv_std:.6f} too small, using raw advantages")
    
    return advs, returns

# ------------------ PPO update ------------------
async def ppo_update():
    global update_steps
    global LAST_UPDATE_METRICS
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

    # Aggregate metrics across all processed batches (across epochs until stop)
    agg_policy_loss = 0.0
    agg_value_loss = 0.0
    agg_entropy = 0.0
    agg_mean_ratio = 0.0
    agg_percent_clipped = 0.0
    agg_batches = 0

    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        epoch_kl = 0.0
        num_batches = 0
        
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

            # Compute KL divergence for early stopping
            with torch.no_grad():
                approx_kl = (oldlp - logp).mean().item()
                epoch_kl += approx_kl
                num_batches += 1

            ratio = torch.exp(logp - oldlp)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # entropy
            probs_food = F.softmax(logits, dim=1)
            entropy_food = -(probs_food * logp_food_all).sum(dim=1).mean()
            entropy_move = move_dist.entropy().sum(dim=1).mean()
            entropy = entropy_food + entropy_move

            # value loss with clipping to stabilize training
            vpred = critic(s)

            # Get old value predictions (aligned with shuffled batch indices)
            old_vpred_t = torch.tensor([values[i] for i in batch], dtype=torch.float32).to(DEVICE)

            # Clipped value loss (similar to policy clipping)
            vpred_clipped = old_vpred_t + torch.clamp(vpred - old_vpred_t, -PPO_CLIP, PPO_CLIP)
            value_loss_unclipped = F.mse_loss(vpred, ret_b)
            value_loss_clipped = F.mse_loss(vpred_clipped, ret_b)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
            
            # Also use Huber loss for robustness
            value_loss_huber = F.smooth_l1_loss(vpred, ret_b)
            value_loss_final = 0.5 * value_loss + 0.5 * value_loss_huber

            # Per-batch stats for aggregation
            with torch.no_grad():
                batch_mean_ratio = ratio.mean().item()
                clipped_mask = ((ratio > (1.0 + PPO_CLIP)) | (ratio < (1.0 - PPO_CLIP))).float()
                batch_percent_clipped = (clipped_mask.mean().item()) * 100.0

            loss = (
                policy_loss
                - (ENTROPY_COEF_MOVE * entropy_move + ENTROPY_COEF_FOOD * entropy_food)
                + VALUE_LOSS_COEF * value_loss_final
            )

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            actor_opt.step()
            critic_opt.step()

            # Accumulate aggregates
            agg_policy_loss += float(policy_loss.item())
            agg_value_loss += float(value_loss_final.item())
            agg_entropy += float(entropy.item())
            agg_mean_ratio += float(batch_mean_ratio)
            agg_percent_clipped += float(batch_percent_clipped)
            agg_batches += 1
        
        # Early stopping based on KL divergence
        avg_kl = epoch_kl / max(num_batches, 1)
        if avg_kl > TARGET_KL:
            print(f"[PPO] Early stopping at epoch {epoch+1}/{PPO_EPOCHS} due to KL={avg_kl:.6f} > {TARGET_KL}")
            break

    update_steps += 1
    
    # Log and persist update metrics (so checkpoint JSON uses same KL)
    # Compute averages across processed batches
    denom = max(agg_batches, 1)
    LAST_UPDATE_METRICS = {
        "policy_loss": float(agg_policy_loss / denom),
        "value_loss": float(agg_value_loss / denom),
        "entropy": float(agg_entropy / denom),
        "approx_kl": float(avg_kl),
        "mean_ratio": float(agg_mean_ratio / denom),
        "percent_clipped": float(agg_percent_clipped / denom),
        "adv_mean": float(advs.mean()),
        "adv_std": float(advs.std()),
        "reward_mean": float(np.mean(rewards) if len(rewards) else 0.0),
        "reward_std": float(np.std(rewards) if len(rewards) else 0.0),
        "return_mean": float(returns.mean() if len(returns) else 0.0),
        "value_mean": float(np.mean(values) if len(values) else 0.0),
    }
    print(
        f"[PPO Update #{update_steps}] "
        f"policy_loss={LAST_UPDATE_METRICS['policy_loss']:.6f}, "
        f"value_loss={LAST_UPDATE_METRICS['value_loss']:.6f}, "
        f"entropy={LAST_UPDATE_METRICS['entropy']:.4f}, "
        f"approx_kl={LAST_UPDATE_METRICS['approx_kl']:.6f}, "
        f"mean_ratio={LAST_UPDATE_METRICS['mean_ratio']:.6f}, percent_clipped={LAST_UPDATE_METRICS['percent_clipped']:.2f}%, "
        f"adv_mean={LAST_UPDATE_METRICS['adv_mean']:.6f}, adv_std={LAST_UPDATE_METRICS['adv_std']:.6f}, "
        f"reward_mean={LAST_UPDATE_METRICS['reward_mean']:.4f}, reward_std={LAST_UPDATE_METRICS['reward_std']:.4f}, "
        f"return_mean={LAST_UPDATE_METRICS['return_mean']:.4f}, value_mean={LAST_UPDATE_METRICS['value_mean']:.4f}"
    )

    # clear buffer
    buffer["states"].clear(); buffer["states_flat"].clear()
    buffer["move_actions"].clear(); buffer["food_actions"].clear()
    buffer["logps"].clear(); buffer["values"].clear()
    buffer["rewards"].clear(); buffer["dones"].clear()
import time
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import math

# Helper numeric conversions
def _to_number(x) -> float:
    if isinstance(x, torch.Tensor):
        try:
            return float(x.detach().cpu().item())
        except Exception:
            return float(x.detach().cpu().numpy().tolist())
    return float(x)

def explained_variance_np(y: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    y = np.array(y, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    denom = np.var(y)
    if denom == 0:
        return None
    return float(1.0 - np.var(y - y_pred) / denom)

def compute_metrics_from_batch(
    states: torch.Tensor,
    move_actions: torch.Tensor,
    food_actions: torch.Tensor,
    old_logps: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    actor,
    critic,
    ppo_clip: float,
    entropy_coef: float,
    device: Optional[torch.device] = None,
    episode_length: Optional[int] = None,
    episode_return: Optional[float] = None,
    max_logp_list: Optional[int] = 1000,
) -> Dict[str, Any]:
    """
    Compute PPO update metrics on a batch (no gradient steps performed).
    Uses diagonal Gaussian for continuous move action; log-prob of (dx,dy) is move_dist.log_prob(a).sum(dim=1).

    Inputs:
      - states: torch.Tensor [N, ...]
      - move_actions: torch.Tensor [N, move_dim]
      - food_actions: torch.Tensor [N] (long)
      - old_logps: torch.Tensor [N] (log probs computed at rollout)
      - returns: torch.Tensor [N]
      - advantages: torch.Tensor [N]
      - actor, critic: callables / nn.Modules
      - ppo_clip: clip epsilon (float)
      - entropy_coef: used only for consistency naming (not required to compute metrics)
      - device: optional torch.device, if provided tensors will be moved
      - episode_length / episode_return: optional episode info to include
      - max_logp_list: if >0 store up to that many per-sample logp_move values in the returned dict

    Returns a dict with keys:
      policy_loss, value_loss, entropy, approx_kl, mean_ratio, percent_clipped,
      explained_variance, adv_mean, adv_std, episode_length, episode_return,
      logp_move_mean, (optionally) logp_move (list)
    """
    if device is not None:
        states = states.to(device)
        move_actions = move_actions.to(device)
        food_actions = food_actions.to(device)
        old_logps = old_logps.to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)

    # Forward pass through policy
    with torch.no_grad():
        mean, logits = actor(states)  # expect mean: [N, move_dim], logits: [N, num_food]
        # move_logstd is expected on actor as a parameter (same as in your actor)
        logstd = actor.move_logstd.unsqueeze(0).expand_as(mean)
        std = torch.exp(logstd)
        move_dist = torch.distributions.Normal(mean, std)  # independent normals per dim

        # log-prob for continuous move action (sum dims to get log p(dx,dy))
        logp_move = move_dist.log_prob(move_actions).sum(dim=1)  # [N]

        # discrete food log-prob
        logp_food_all = F.log_softmax(logits, dim=1)
        logp_food = logp_food_all[torch.arange(len(food_actions)), food_actions]

        logp = logp_move + logp_food  # total log-prob per sample

        # ratios
        ratio = torch.exp(logp - old_logps)

        # surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # entropy: continuous + discrete
        entropy_move = move_dist.entropy().sum(dim=1).mean()
        probs_food = F.softmax(logits, dim=1)
        entropy_food = -(probs_food * logp_food_all).sum(dim=1).mean()
        entropy = entropy_move + entropy_food

        # value predictions for value loss
        vpred = critic(states).squeeze(-1) if critic(states).dim() > 1 else critic(states)
        value_loss = F.mse_loss(vpred, returns)

        # approx_kl
        approx_kl = float(torch.mean(old_logps - logp).item())
        mean_ratio = float(ratio.mean().item())
        clipped_mask = ((ratio > (1.0 + ppo_clip)) | (ratio < (1.0 - ppo_clip))).float()
        percent_clipped = float(clipped_mask.mean().item()) * 100.0

        # explained variance: compare returns (numpy) vs vpred (numpy)
        try:
            ev = explained_variance_np(returns.detach().cpu().numpy(), vpred.detach().cpu().numpy())
        except Exception:
            ev = None

        # adv stats
        adv_np = advantages.detach().cpu().numpy()
        adv_mean = float(np.mean(adv_np)) if adv_np.size > 0 else 0.0
        adv_std = float(np.std(adv_np, ddof=0)) if adv_np.size > 0 else 0.0

        # logp_move stats and sample list (cap to max_logp_list to avoid huge JSON)
        logp_move_np = logp_move.detach().cpu().numpy()
        logp_move_mean = float(np.mean(logp_move_np)) if logp_move_np.size > 0 else 0.0
        logp_move_list = logp_move_np.tolist()[:max_logp_list] if max_logp_list and logp_move_np.size > 0 else []

    metrics: Dict[str, Any] = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "percent_clipped": percent_clipped,
        "explained_variance": ev,
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "episode_length": episode_length,
        "episode_return": episode_return,
        "logp_move_mean": logp_move_mean,
        "logp_move": logp_move_list,
    }
    return metrics


# Updated save_checkpoint that accepts metrics (and will use compute_metrics_from_batch if given raw batch)
async def save_checkpoint(
    ep: Optional[int] = None,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    auto_compute_from_buffer: bool = True,
    max_logp_list: int = 1000,
) -> Tuple[Path, Optional[Path]]:
    """
    Save model checkpoint (.pt) and (when ep is provided) a JSON summary file.
    If metrics is None and auto_compute_from_buffer is True, try to compute metrics from global buffer snapshot.
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
            "timestamp": ts,
        }
    except NameError as e:
        raise RuntimeError(
            "Missing required training objects. Ensure 'actor', 'critic', 'actor_opt', "
            "'critic_opt' and 'update_steps' are defined in scope."
        ) from e

    # Save torch checkpoint off the event loop
    await asyncio.to_thread(torch.save, checkpoint, str(pt_fname))
    print(f"[PPO] Saved checkpoint: {pt_fname}")

    # If metrics not provided and auto_compute_from_buffer is requested, try to build them
    if metrics is None and auto_compute_from_buffer:
        try:
            # snapshot relevant buffers (do not mutate)
            print("[PPO] Auto-computing metrics from buffer snapshot...")
            states = torch.stack([torch.from_numpy(s) for s in buffer["states_flat"]]).to(actor.move_logstd.device)
            move_actions = torch.tensor(buffer["move_actions"], dtype=torch.float32).to(actor.move_logstd.device)
            food_actions = torch.tensor(buffer["food_actions"], dtype=torch.long).to(actor.move_logstd.device)
            old_logps = torch.tensor(buffer["logps"], dtype=torch.float32).to(actor.move_logstd.device)
            # compute GAE if needed: assume buffer["values"] and buffer["rewards"], buffer["dones"] exist
            try:
                advs, returns = compute_gae_advantages(buffer["rewards"], buffer["values"], buffer["dones"],
                                                      gamma=globals().get("GAMMA", 0.92),
                                                      lam=globals().get("GAE_LAMBDA", 0.95))
                advs_t = torch.from_numpy(advs).to(actor.move_logstd.device)
                returns_t = torch.from_numpy(returns).to(actor.move_logstd.device)
            except Exception:
                advs_t = torch.tensor(buffer.get("advantages", []), dtype=torch.float32).to(actor.move_logstd.device)
                returns_t = torch.tensor(buffer.get("returns", []), dtype=torch.float32).to(actor.move_logstd.device)
            metrics = compute_metrics_from_batch(
                states=states,
                move_actions=move_actions,
                food_actions=food_actions,
                old_logps=old_logps,
                returns=returns_t,
                advantages=advs_t,
                actor=actor,
                critic=critic,
                ppo_clip=globals().get("PPO_CLIP", 0.2),
                entropy_coef=globals().get("ENTROPY_COEF", 0.01),
                device=actor.move_logstd.device,
                episode_length=globals().get("last_episode_length", None),
                episode_return=globals().get("last_episode_return", None),
                max_logp_list=max_logp_list,
            )
        except Exception as exc:
            print(f"[PPO] Warning: failed to auto-compute metrics from buffer: {exc}")
            metrics = None

    # Use provided metrics or last-known metrics
    metrics = metrics or globals().get("LAST_UPDATE_METRICS") or {}

    # Normalize metrics for JSON and printing
    printable_metrics: Dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            # convert tensors to numbers/lists
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    printable_metrics[k] = _to_number(v)
                else:
                    printable_metrics[k] = v.detach().cpu().tolist()
            else:
                printable_metrics[k] = v
        except Exception:
            printable_metrics[k] = str(v)

    # Print summary line for key metrics if present
    order = [
        "policy_loss", "value_loss", "entropy", "approx_kl", "mean_ratio", "percent_clipped",
        "explained_variance", "adv_mean", "adv_std", "episode_length", "episode_return", "logp_move_mean"
    ]
    parts = []
    for k in order:
        if k in printable_metrics and printable_metrics[k] is not None:
            val = printable_metrics[k]
            if isinstance(val, (float, int)):
                parts.append(f"{k}={val:.6g}")
            else:
                parts.append(f"{k}={val}")
    if parts:
        print("[PPO][Checkpoint Metrics] " + ", ".join(parts))

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
            rr_list = list(rr)
            recent_avg_return = float(sum(rr_list) / len(rr_list)) if len(rr_list) > 0 else 0.0
    except Exception:
        recent_avg_return = None

    json_obj = {
        "timestamp": ts,
        "episode": ep,
        "checkpoint": str(pt_fname.name),
        "config": config,
        "stats": stats,
        "recent_avg_return": recent_avg_return,
        "metrics": printable_metrics,
    }

    json_fname = log_dir / f"episode_{ep}_{ts}.json" if ep is not None else None

    if json_fname is not None:
        def _write_json(obj, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

        try:
            await asyncio.to_thread(_write_json, json_obj, json_fname)
            print(f"[PPO] Saved JSON summary: {json_fname}")
        except Exception as exc:
            print(f"[PPO] Warning: failed to save JSON summary {json_fname}: {exc}")

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
                # perform PPO update first so metrics reflect actual update-time values
                await ppo_update()
                # Save checkpoint with the latest update metrics so JSON reflects early-stopping KL
                try:
                    await save_checkpoint(episodes, metrics=LAST_UPDATE_METRICS, auto_compute_from_buffer=False)
                except Exception:
                    await save_checkpoint(episodes)
                episodes += 1
                games_completed = 0
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
            
            # Epsilon-greedy for food action (helps maintain diversity)
            if USE_FOOD_EPSILON and np.random.rand() < FOOD_EPSILON:
                food_sample = int(np.random.randint(0, len(FOOD_ACTIONS)))
            else:
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
        # Normalize reward for more stable training
        normalized_reward = normalize_reward(float(reward))
        buffer['rewards'].append(normalized_reward)
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
