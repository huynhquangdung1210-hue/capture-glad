"""Headless backend game simulator (async) to train `RL_PPO.py`.

Usage:
  python src/backend_game.py

This script connects to ws://localhost:8765 and acts like the browser game
client: it sends per-prey transition messages and receives actions back
from the RL server (PPO or Q-learning). It runs simple physics and game rules 
(stars, predators, special, reproduce, starvation) sufficient for training the RL server.

Requirements:
  pip install websockets numpy
"""
import asyncio
import json
import math
import random
import time
import uuid
import math
import websockets
import numpy as np
import math


# ================================================
# CONFIGURATION
# ================================================

# WebSocket connection to RL server
WS_URI = "ws://localhost:8765"

# Simulation parameters
TICK_DT = 1.0 / 20.0        # Simulation timestep: 20 ticks/second
MAX_GAME_TIME = 120.0       # Maximum game duration in seconds

# Initial entity counts
INITIAL_PREY = 12           # Number of prey at game start
INITIAL_PREDATORS = 7       # Number of predators at game start
INITIAL_STARS = 60          # Number of food stars at game start

# Arena configuration
ARENA_RADIUS = 300          # Circular arena radius in pixels

# Movement speeds (pixels per second)
PREY_SPEED = 6.6           # Prey movement speed
PREDATOR_SPEED = 5.4       # Predator movement speed

# Interaction distances
PREDATOR_EAT_RANGE = 12    # Distance at which predator catches prey
STAR_EAT_RANGE = 10        # Distance at which prey collects star
SPECIAL_KILL_RADIUS = 25   # Radius of prey special ability effect

# Energy costs for prey abilities
SPECIAL_COST = 2           # Energy cost to activate special ability
REPRODUCE_COST = 2         # Energy cost to reproduce

# ================================================
# REWARD SHAPING
# ================================================
# These weights define the reward function for RL training.
# Should match client-side rewards for consistency.

RL_WEIGHTS = {
    "specialeff": 50,      # Reward per predator killed with special ability
    "specialwaste": -1,    # Penalty per tick when special is active but unused
                           # (20 ticks/sec × 3 sec = -60 total if no kills)
    "starve": -25,         # Large penalty for death by starvation or predator
    "foodgain": 20,        # Reward for collecting a star
    "survival": 0.5,       # Small reward per tick for staying alive
    "reproduce": 50,       # Large reward for successful reproduction
    "outbound": -1         # Penalty for going outside arena bounds
}


# ================================================
# UTILITY FUNCTIONS
# ================================================

def dist(a, b):
    """Calculate Euclidean distance between two points.
    
    Args:
        a: Tuple (x, y) for first point
        b: Tuple (x, y) for second point
    
    Returns:
        Distance as float
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize(dx, dy):
    """Normalize a direction vector to unit length.
    
    Args:
        dx: X component of direction
        dy: Y component of direction
    
    Returns:
        Tuple (nx, ny) with length 1, or (0, 0) if input is zero vector
    """
    m = math.hypot(dx, dy)
    if m == 0:
        return 0.0, 0.0
    return dx / m, dy / m


def rnd_pos(radius=ARENA_RADIUS):
    """Generate random position within arena bounds.
    
    Args:
        radius: Arena radius (default: ARENA_RADIUS)
    
    Returns:
        Tuple (x, y) with coordinates in [-radius, radius]
    """
    return (random.uniform(-radius, radius), random.uniform(-radius, radius))


# ================================================
# ENTITY CLASSES
# ================================================

class Prey:
    """Prey entity with energy, special ability, and RL state tracking.
    
    Prey must collect stars to maintain energy and avoid starvation.
    They can activate a special ability that makes them invincible and
    allows them to kill nearby predators.
    """
    
    def __init__(self):
        # Unique identifier
        self.id = "prey-" + uuid.uuid4().hex[:8]
        
        # Position and velocity
        self.x, self.y = rnd_pos()
        self.vx = 0.0
        self.vy = 0.0
        
        # Energy and survival mechanics
        self.energy = 3              # Energy level (decreases over time)
        self.starvationTimer = 0.0   # Counts up to 5s, then energy decreases
        
        # Special ability state
        self.special = 0.0           # Seconds remaining of special ability
        
        # RL training variables
        self.rlReward = 0.0          # Accumulated reward for this timestep
        self.prev_state = None       # Previous state for TD learning
        self.prev_action = None      # Previous action for TD learning
        
        # Status flags
        self.justeat = False         # True if just collected a star
        self.alive = True            # False when killed or starved
        self.terminal = False        # True when episode ends for this prey
    
    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            'energy':self.energy,
            'special':self.special,
            'starvationtimer':self.starvationTimer


        }

class Predator:
    def __init__(self):
        self.id = "pred-" + uuid.uuid4().hex[:8]
        self.x, self.y = rnd_pos()
        self.vx = 0.0
        self.vy = 0.0
    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y
        }

class Star:
    def __init__(self):
        self.id = "star-" + uuid.uuid4().hex[:8]
        self.x, self.y = rnd_pos()
    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y
        }
# ================================================
# STATE REPRESENTATION
# ================================================

def stars_sort(stars, prey):
    """Sort stars by distance from prey (nearest first).
    
    RL agents receive sorted lists so they can easily access the
    nearest food source without complex spatial reasoning.
    
    Args:
        stars: List of Star objects
        prey: Prey object to measure distance from
    
    Returns:
        Sorted list of stars, or None if empty
    """
    if not stars:
        return None
    return sorted(
        stars,
        key=lambda s: math.hypot(s.x - prey.x, s.y - prey.y)
    )

def predators_sort(predators, prey):
    """Sort predators by distance from prey (nearest first).
    
    Args:
        predators: List of Predator objects
        prey: Prey object to measure distance from
    
    Returns:
        Sorted list of predators, or None if empty
    """
    if not predators:
        return None
    return sorted(
        predators,
        key=lambda p: math.hypot(p.x - prey.x, p.y - prey.y)
    )


def build_state(prey, stars, predators):
    """Construct state representation for RL agent.
    
    State format: [prey_info, sorted_stars_list, sorted_predators_list]
    All entities are converted to dictionaries and sorted by distance.
    
    Args:
        prey: Prey object
        stars: List of Star objects
        predators: List of Predator objects
    
    Returns:
        List with prey dict and sorted lists of stars/predators
    """
    # Sort entities by distance to prey (nearest first)
    sorted_stars_obj = stars_sort(stars, prey)
    sorted_stars = [star.to_dict() for star in sorted_stars_obj] if sorted_stars_obj else None
    
    sorted_predators_obj = predators_sort(predators, prey)
    sorted_predators = [pred.to_dict() for pred in sorted_predators_obj] if sorted_predators_obj else None
    
    prey_reduced = prey.to_dict()

    return [prey_reduced, sorted_stars, sorted_predators]

def keep_inside(e, arena_radius=ARENA_RADIUS):
    """Keep entity within circular arena bounds.
    
    If entity exceeds arena radius, it is pushed back to the boundary
    and penalized (if it has rlReward attribute).
    
    Args:
        e: Entity object with x, y coordinates (and optionally rlReward)
        arena_radius: Maximum distance from origin
    """
    d = math.sqrt(e.x**2 + e.y**2)
    if d > arena_radius:
        # Apply penalty for leaving arena
        if hasattr(e, 'rlReward'):
            e.rlReward += RL_WEIGHTS.get("outbound")
        
        # Scale position back to arena boundary
        scale = arena_radius / d
        e.x *= scale
        e.y *= scale

async def run_game_loop(ws, game_index=0):
    # initialize world
    preys = [Prey() for _ in range(INITIAL_PREY)]
    preds = [Predator() for _ in range(INITIAL_PREDATORS)]
    stars = [Star() for _ in range(INITIAL_STARS)]
    t = 0.0
    tick = 0

    while t < MAX_GAME_TIME and any(p.alive for p in preys):
        # for each prey, send transition and await action
        # print(f"Tick {tick} t={t:.2f}s")
        # total_reward = []
        for p in list(preys):
            if not p.alive and not p.terminal:
                continue
            if not p.alive and p.terminal:
                msg = {
                "state": state,
                "terminal": True,
                "reward": p.rlReward or 0.0,
                "prev_state": p.prev_state if p.prev_state is not None else None,
                "prev_action": p.prev_action if p.prev_action is not None else None,
                "preyId": p.id,
                }
                await ws.send(json.dumps(msg))
                p.rlReward = 0.0
                p.terminal= False
                continue
            state = build_state(p, stars, preds)
            # send transition to server
            msg = {
                "state": state,
                "terminal": False,
                "reward": p.rlReward or 0.0,
                "prev_state": p.prev_state if p.prev_state is not None else None,
                "prev_action": p.prev_action if p.prev_action is not None else None,
                "preyId": p.id,
            }
            await ws.send(json.dumps(msg))

            # reset reward after sending (client mirrors behavior)
            p.rlReward = 0.0

            # await server response for this prey
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                # if server doesn't respond in time, skip
                continue

            try:
                data = json.loads(resp)
            except Exception:
                continue

            if data.get("preyId") != p.id:
                # not our reply (could be interleaved), try to proceed
                pass

            dx = data.get("dx", 0.0)
            dy = data.get("dy", 0.0)
            action_idx = data.get("action_idx")
            foodAction = data.get("foodAction")

            # apply foodAction
            if foodAction == "ACTIVATE_SPECIAL":
                if p.energy >= SPECIAL_COST:
                    # print("Activating special for prey")
                    p.special = 3.0  # active seconds
                    p.energy -= SPECIAL_COST
                    # print(p.rlReward)

            elif foodAction == "REPRODUCE":
                if p.energy >= REPRODUCE_COST:
                    p.energy -= REPRODUCE_COST
                    baby = Prey()
                    baby.x, baby.y = p.x + 5, p.y + 5
                    preys.append(baby)
                    # reward for reproducing
                    p.rlReward += RL_WEIGHTS.get("reproduce")
                    # print("reproduce", p.rlReward)

            # movement: set velocity based on dx,dy normalized
            nx, ny = normalize(dx, dy)
            p.vx = nx * PREY_SPEED
            p.vy = ny * PREY_SPEED

            # save prev info for next tick
            p.prev_state = state
            p.prev_action = action_idx
        # -- physics integration for all entities
        # move preys
        for p in list(preys):
            if not p.alive:
                continue
            p.rlReward += RL_WEIGHTS.get("survival") * TICK_DT
            p.x += p.vx * TICK_DT
            p.y += p.vy * TICK_DT
            # update special timer
            if p.special > 0:
                p.special = max(0.0, p.special - TICK_DT)

            # check star eating
            for s in list(stars):
                if dist((p.x, p.y), (s.x, s.y)) < STAR_EAT_RANGE:
                    p.energy = (p.energy or 0) + 1
                    p.justeat = True
                    p.rlReward += RL_WEIGHTS.get("foodgain")
                    try:
                        stars.remove(s)
                    except Exception:
                        pass
                    break

            # predators eat prey
            num_pred_killed=0

            for pr in list(preds):
                                    # if prey has special active, predator is killed instead
                if p.special > 0 and dist((p.x, p.y), (pr.x, pr.y)) < SPECIAL_KILL_RADIUS:
                    try:
                        preds.remove(pr)
                    except Exception:
                        pass
                    num_pred_killed+=1
                    p.rlReward += RL_WEIGHTS.get("specialeff")
                    # print("specialeff",p.rlReward)
                elif dist((p.x, p.y), (pr.x, pr.y)) < PREDATOR_EAT_RANGE:
                    p.alive = False
                    p.terminal = True
                    p.rlReward += RL_WEIGHTS.get("starve")
                    break
            if num_pred_killed==0 and p.special > 0:
                p.rlReward += RL_WEIGHTS.get("specialwaste")
                # print("specialwaste",p.rlReward)
            # starvation timer & energy decay every 5s
            p.starvationTimer += TICK_DT
            if p.starvationTimer >= 5.0:
                p.energy = (p.energy or 0) - 1
                p.starvationTimer = 0.0

            if p.energy <= -1:
                p.alive = False
                p.rlReward += RL_WEIGHTS.get("starve")
            keep_inside(p)

            # total_reward.append(p.rlReward)
        # print(" Total prey reward this tick:",total_reward)


        # move predators (simple chase nearest prey or wander)
        for prd in preds:
            alive_preys = [pp for pp in preys if pp.alive]
            if not alive_preys:
                prd.vx, prd.vy = normalize(random.uniform(-1, 1), random.uniform(-1, 1))
            else:
                target = min(alive_preys, key=lambda p: dist((p.x, p.y), (prd.x, prd.y)))
                dx = target.x - prd.x
                dy = target.y - prd.y
                nx, ny = normalize(dx, dy)
                prd.vx = nx * PREDATOR_SPEED
                prd.vy = ny * PREDATOR_SPEED
            prd.x += prd.vx * TICK_DT
            prd.y += prd.vy * TICK_DT

        # respawn stars gradually
        while len(stars) < INITIAL_STARS:
            stars.append(Star())
        while len(stars) < INITIAL_PREDATORS:
            preds.append(Predator())
        # advance time
        await asyncio.sleep(TICK_DT)
        t += TICK_DT
        tick += 1

    # game finished
    total_reward = sum((p.rlReward or 0.0) for p in preys)
    end_msg = {"type": "game_end", "elapsed": t, "preyCount": sum(1 for p in preys if p.alive), "totalReward": total_reward}
    # print(f"Game #{game_index} ended: time={t:.2f}s, surviving prey={end_msg['preyCount']}, totalReward={total_reward:.2f}")    
    try:
        await ws.send(json.dumps(end_msg))
    except Exception as e:
        print("Failed to notify server of game end:", e)


async def main():
    print("Connecting to RL server at", WS_URI)
    async with websockets.connect(WS_URI) as ws:
        game_idx = 0
        while True:
            print(f"Starting game #{game_idx + 1}")
            try:
                await run_game_loop(ws, game_idx)
            except websockets.ConnectionClosed:
                print("Connection closed by server")
                break
            except Exception as e:
                print("Error during game loop:", e)
                break
            game_idx += 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")