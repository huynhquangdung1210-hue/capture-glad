# PPO Training Fixes - Addressing Training Collapse

## Problem Diagnosis

The initial training metrics indicated severe training issues:

- **policy_loss = -7.40352e-07**: Near-zero, indicating the policy was not learning
- **value_loss = 0.270569**: Poor value network predictions
- **entropy = 9.60289**: Extremely high, suggesting the policy remained random
- **approx_kl = 2.00774e-07**: Near-zero KL divergence, policy barely changing
- **mean_ratio = 1**: No policy updates taking effect
- **percent_clipped = 0**: No clipping happening, policy effectively frozen
- **explained_variance = -143.673**: Value network worse than predicting the mean
- **adv_mean = 6.77611e-07, adv_std = 1**: Advantages normalized to pure noise
- **Average return decreasing linearly**: Agent performance degrading over time

## Root Causes

1. **Value Network Collapse**: The critic couldn't predict returns accurately, causing GAE to produce meaningless advantages
2. **Vanishing Gradients**: Reward scale was too small or learning rate too low for effective updates
3. **Advantage Normalization Issues**: With poor value estimates, normalized advantages became pure noise
4. **Training Instability**: Policy and value networks were learning at mismatched rates

## Applied Fixes

### 1. Hyperparameter Adjustments

```python
LEARNING_RATE = 1e-3              # Increased from 3e-4 for faster learning
CRITIC_LEARNING_RATE = 3e-3       # NEW: Higher LR for critic to catch up
GAMMA = 0.98                      # Increased from 0.92 to value long-term more
GAE_LAMBDA = 0.90                 # Reduced from 0.95 to reduce bias
PPO_EPOCHS = 3                    # Reduced from 4 to prevent overfitting
ENTROPY_COEF = 0.005              # Reduced from 0.01 to encourage exploitation
VALUE_LOSS_COEF = 1.0             # NEW: Increased from 0.5 to prioritize value learning
MAX_GRAD_NORM = 1.0               # Increased from 0.5 for larger updates
TARGET_KL = 0.015                 # NEW: Early stopping threshold
```

### 2. Reward Normalization

Added running reward normalization to stabilize training:
- Tracks running mean and std of rewards
- Normalizes rewards before storing in buffer
- Clips normalized rewards to [-10, 10]
- Prevents reward scale issues

### 3. Improved Advantage Computation

```python
# More robust advantage normalization
if len(advs) > 1:
    adv_std = advs.std()
    if adv_std > 0.01:  # Only normalize if std is reasonable
        advs = (advs - adv_mean) / (adv_std + 1e-8)
    else:
        advs = advs - adv_mean  # Just center if too uniform
```

### 4. Value Loss Improvements

**Clipped Value Loss** (prevents large value updates):
```python
vpred_clipped = old_vpred + torch.clamp(vpred - old_vpred, -PPO_CLIP, PPO_CLIP)
value_loss_unclipped = F.mse_loss(vpred, ret_b)
value_loss_clipped = F.mse_loss(vpred_clipped, ret_b)
value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
```

**Huber Loss** (robust to outliers):
```python
value_loss_huber = F.smooth_l1_loss(vpred, ret_b)
value_loss_final = 0.5 * value_loss + 0.5 * value_loss_huber
```

### 5. Early Stopping on KL Divergence

Prevents policy from deviating too much from old policy:
```python
avg_kl = epoch_kl / max(num_batches, 1)
if avg_kl > TARGET_KL:
    print(f"Early stopping at epoch {epoch+1} due to KL={avg_kl:.6f}")
    break
```

### 6. Better Initialization

```python
# Initialize log_std to -0.5 (std ≈ 0.6) for reasonable exploration
self.move_logstd = nn.Parameter(torch.ones(2) * -0.5)
```

### 7. Enhanced Logging

Added detailed logging to track:
- Policy loss, value loss, entropy
- KL divergence, advantage statistics
- Reward statistics (mean, std)
- Return and value predictions
- Periodic reward normalization stats

## Expected Improvements

After these changes, you should see:
1. **Higher policy loss** (e.g., 0.01-0.1): Policy actively learning
2. **Decreasing value loss**: Critic learning to predict returns
3. **Lower entropy** over time: Policy becoming more deterministic
4. **KL divergence** in range 0.001-0.02: Policy updating meaningfully
5. **Explained variance** increasing toward 0-1: Value predictions improving
6. **Advantages** with meaningful std (not 1.0): Real signal from environment
7. **Average return** stabilizing or increasing: Agent improving

## Monitoring Tips

Watch for these in logs:
- If KL consistently hits target early: Reduce `TARGET_KL` or increase `LEARNING_RATE`
- If value loss stays high: Increase `CRITIC_LEARNING_RATE` or `VALUE_LOSS_COEF`
- If entropy drops too fast: Increase `ENTROPY_COEF`
- If rewards vary wildly: Check `REWARD_NORM_CLIP` in normalization logs

## Next Steps if Still Issues

1. **Check reward signal**: Is environment giving meaningful rewards?
2. **Verify state representation**: Are inputs normalized/scaled properly?
3. **Inspect trajectories**: Are episodes too short/long?
4. **Add reward shaping**: Consider dense rewards for learning
5. **Adjust UPDATE_INTERVAL**: Try 1024 or 4096 if still unstable
