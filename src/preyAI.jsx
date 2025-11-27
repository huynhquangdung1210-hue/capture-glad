/**
 * RL Prey AI System with starvation mechanics and reward calculation.
 * 
 * This module manages prey behavior including:
 * - AI decision making (heuristics vs RL)
 * - State preparation for RL agents
 * - Reward calculation based on survival, reproduction, and special abilities
 * - Action registration and execution
 */

import { length, randRange } from "./utils.jsx";
import { rlSocket } from "./rlConnection.jsx";

// ================================================
// AI TYPE CONFIGURATION
// ================================================

/**
 * Available AI modes for prey behavior.
 */
export const AIType = {
  HEURISTICS: "heuristics",  // Rule-based decision making
  RL: "rl",                  // Reinforcement learning (neural network)
};

// ================================================
// REWARD STRUCTURE FOR RL TRAINING
// ================================================

/**
 * Reward weights that shape the RL agent's learning.
 * These values define what behaviors are encouraged or discouraged.
 */
export const RL_WEIGHTS = {
  specialeff: 7,        // Reward per predator killed with special ability
  specialwaste: -10,    // Penalty when special ability is wasted (no kills)
  starve: -20,          // Large penalty for dying (starvation or caught)
  foodgain: 1.0,        // Reward for collecting a star
  survival: 0.01,       // Small reward per timestep for staying alive
  reproduce: 5.0        // Reward for successful reproduction
};

// ================================================
// PREY ABILITY PARAMETERS
// ================================================

/**
 * Configuration for prey abilities and mechanics.
 */
export const PREY_STATS = {
  special_cost: 5,             // Energy cost to activate special ability
  reproduce_cost: 5,           // Energy cost to reproduce
  specialkillradius: 50,       // Radius (pixels) in which special kills predators
  specialtime: 60,             // Duration (ticks) special ability lasts
  nearbypredatorcutoff: 100    // Distance threshold for "predator nearby" detection
};


// RL Action sets
export const RL_ACTIONS = {
  MOVE: ["PURSUIT_FOOD", "RUN_FROM_PREDATOR"],
  FOOD: ["REPRODUCE", "ACTIVATE_SPECIAL", "IDLE_FOOD"]
};

// Map: preyId -> { dx, dy, moveAction, foodAction, action_idx }
const rlActions = new Map();

// Map: preyId -> { prev_state, prev_action }
const rlPreyMap = new Map();


// ================================================
// STATE PREPARATION UTILITIES
// ================================================

/**
 * Sort stars by distance from prey (nearest first).
 * 
 * RL agents benefit from sorted lists so they can easily
 * identify the nearest food source without complex reasoning.
 * 
 * @param {Array} stars - List of star objects with x, y positions
 * @param {Object} prey - Prey object with x, y position
 * @returns {Array|null} Sorted array of stars, or null if empty
 */
function stars_sort(stars, prey) {
  if (stars.length === 0) {
    return null;
  }
  
  const sorted = [...stars].sort((a, b) => {
    const da = Math.hypot(a.x - prey.x, a.y - prey.y);
    const db = Math.hypot(b.x - prey.x, b.y - prey.y);
    return da - db;  // Ascending order (nearest first)
  });
  
  return sorted;
}

/**
 * Sort predators by distance from prey (nearest first).
 * 
 * Helps prey identify immediate threats.
 * 
 * @param {Array} predators - List of predator objects
 * @param {Object} prey - Prey object with x, y position
 * @returns {Array|null} Sorted array of predators, or null if empty
 */
function predators_sort(predators, prey) {
  if (predators.length === 0) {
    return null;
  }
  
  const sorted = [...predators].sort((a, b) => {
    const da = Math.hypot(a.x - prey.x, a.y - prey.y);
    const db = Math.hypot(b.x - prey.x, b.y - prey.y);
    return da - db;
  });
  
  return sorted;
}
/**
 * Convert object to plain dictionary, removing functions.
 * 
 * This ensures state can be serialized to JSON for sending to Python server.
 * 
 * @param {Object} obj - Object to convert
 * @returns {Object} Plain object without functions
 */
function toDict(obj) {
  if (obj === null || typeof obj !== "object") return obj;

  const out = {};
  for (const key in obj) {
    if (!Object.prototype.hasOwnProperty.call(obj, key)) continue;

    const val = obj[key];

    if (typeof val === "object") {
      out[key] = toDict(val);   // Recursively convert nested objects
    } else if (typeof val !== "function") {
      out[key] = val;           // Copy non-function values
    }
  }
  return out;
}

/**
 * Build RL state representation from game entities.
 * 
 * Creates a structured state that the RL agent can process:
 * - Prey information (position, energy, special ability status)
 * - Sorted list of nearest stars (food sources)
 * - Sorted list of nearest predators (threats)
 * 
 * All objects are converted to plain dictionaries for JSON serialization.
 * 
 * @param {Object} prey - Prey entity
 * @param {Array} stars - Array of star entities
 * @param {Array} predators - Array of predator entities
 * @returns {Array} State as [prey_dict, stars_array, predators_array]
 */
export function getRLState(prey, stars, predators) {
  // Sort and convert stars
  const sorted_stars_obj = stars_sort(stars, prey);
  const sorted_stars = sorted_stars_obj ? sorted_stars_obj.map(toDict) : null;

  // Sort and convert predators
  const sorted_predators_obj = predators_sort(predators, prey);
  const sorted_predators = sorted_predators_obj ? sorted_predators_obj.map(toDict) : null;

  // Convert prey to dict
  const prey_reduced = toDict(prey);
  
  return [
    prey_reduced,  
    sorted_stars,
    sorted_predators,
  ];
}

/* ================================================
   ================================================
   Rule-based decision making for prey when not using RL.
   
   Strategy:
   1. Always move towards nearest star (food)
   2. Activate special ability if predator is nearby and have enough energy
   3. Reproduce if have excess energy and no immediate threat
*/

/**
 * Heuristic (rule-based) prey decision making.
 * 
 * This provides a baseline behavior without machine learning.
 * The agent uses simple rules to survive:
 * - Move towards food
 * - Use special ability when threatened
 * - Reproduce when safe
 * 
 * @param {Object} prey - Prey entity
 * @param {Array} stars - Array of star entities
 * @param {Array} predators - Array of predator entities
 * @returns {Object} Action with {dx, dy, foodAction}
 */
function heuristicPrey(prey, stars, predators) {
  // Sort entities by distance
  const sorted_stars = stars_sort(stars, prey);
  const sorted_predators = predators_sort(predators, prey);
  
  // Check if predator is dangerously close
  const prednearby = (sorted_predators && sorted_predators.length > 0)
    ? (Math.hypot(prey.x - sorted_predators[0].x, prey.y - sorted_predators[0].y) < PREY_STATS.nearbypredatorcutoff)
    : false;
  
  // Decide on food/ability action
  let food_action = "IDLE_FOOD";  // Default: no special action
  
  if (prednearby && prey.special == false && prey.energy >= PREY_STATS.special_cost) {
    // Threat detected: activate special ability
    food_action = "ACTIVATE_SPECIAL";
  } else if (prey.energy >= PREY_STATS.reproduce_cost) {
    // Safe and have energy: reproduce
    food_action = "REPRODUCE";
  }
  
  // If no stars available, move randomly
  if (!sorted_stars) { 
    return { dx: randRange(-1, 1), dy: randRange(-1, 1), foodAction: food_action };
  }
  
  // Move towards nearest star
  const dx = sorted_stars[0].x - prey.x;
  const dy = sorted_stars[0].y - prey.y;

  return { dx, dy, food_action, foodAction: food_action };
}

/* ================================================
   RL PREY LOGIC
   ================================================
   Reinforcement learning-based decision making.
   
   This communicates with a Python RL server via WebSocket:
   1. Build and send current state
   2. Send accumulated reward and previous transition
   3. Receive action from trained model
   4. Execute action in game
   5. Fallback to heuristics if no action received
*/

/**
 * RL-based prey decision making.
 * 
 * Sends state to Python RL server and receives actions from the trained model.
 * Uses WebSocket for real-time communication during training or inference.
 * 
 * @param {Object} prey - Prey entity
 * @param {Array} stars - Array of star entities
 * @param {Array} predators - Array of predator entities
 * @param {Function} getRLState - Function to build state representation
 * @returns {Object} Action with {dx, dy, foodAction}
 */
function rlPrey(prey, stars, predators, getRLState) {
  // ================================================
  // 1. BUILD STATE REPRESENTATION
  // ================================================
  const state = getRLState(prey, stars, predators);
  prey.rlPrevState = state[0];  // Save for next transition
  const prev = rlPreyMap.get(prey.id) || {};
  
  // ================================================
  // 2. SEND TRANSITION TO RL SERVER
  // ================================================
  // Format: (prev_state, prev_action, reward, new_state)
  // This is the SARS tuple for temporal difference learning
  if (rlSocket.readyState === WebSocket.OPEN) {
    rlSocket.send(JSON.stringify({
      state,                               // Current state s_t
      reward: prey.rlReward || 0,         // Reward r_t
      prev_state: prev.prev_state || null,// Previous state s_{t-1}
      prev_action: prev.prev_action || null, // Previous action a_{t-1}
      preyId: prey.id                     // Unique identifier
    }));
  }

  // Reset reward after sending (will accumulate again next timestep)
  prey.rlReward = 0;

  // --------------------------
  // 3. If Python RL returned an action, use it
  // --------------------------
  const action = rlActions.get(prey.id);
  if (action) {
    // remove it from buffer
    rlActions.delete(prey.id);

    // save transition info for next step
    rlPreyMap.set(prey.id, {
      prev_state: prey.rlPrevState,
      prev_action: action.action_idx
    });

    // Return full movement + foodAction
    return {
      dx: action.dx,
      dy: action.dy,
      foodAction: action.foodAction || "IDLE_FOOD"
    };
  }

  // --------------------------
  // 4. Otherwise fallback to heuristics
  // --------------------------
  return heuristicPrey(prey, stars, predators);  // must return {dx, dy, foodAction}
}

/* ================================================
   PREY AI FACTORY
   ================================================
   Creates prey AI controller with specified behavior type.
   Provides unified interface for both heuristic and RL agents.
*/

/**
 * Factory function to create a prey AI controller.
 * 
 * This provides a consistent interface regardless of AI type,
 * making it easy to switch between heuristics and RL during
 * development or comparison experiments.
 * 
 * @param {Object} config - Configuration object
 * @param {string} config.type - AI type (AIType.HEURISTICS or AIType.RL)
 * @param {Function} config.getRLState - State builder function (required for RL)
 * @returns {Object} AI controller with getAction method
 */
export function createPreyAI({ type = AIType.HEURISTICS, getRLState = null } = {}) {
  return {
    /**
     * Get action for a prey based on current game state.
     * 
     * @param {Object} prey - Prey entity
     * @param {Array} stars - Star entities
     * @param {Array} predators - Predator entities
     * @returns {Object} Action {dx, dy, foodAction}
     */
    getAction(prey, stars, predators) {
      if (type === AIType.HEURISTICS) {
        return heuristicPrey(prey, stars, predators);
      }
      if (type === AIType.RL) {
        return rlPrey(prey, stars, predators, getRLState);
      }
      // Default fallback to heuristics
      return heuristicPrey(prey, stars, predators);
    }
  };
}

/* -------------------------------------
   Python → JS: register RL action
--------------------------------------*/
export function registerRLAction(preyId, dx, dy, action_idx, moveAction, foodAction) {
  rlActions.set(preyId, { dx, dy, action_idx, moveAction, foodAction });

  const prev = rlPreyMap.get(preyId) || {};
  rlPreyMap.set(preyId, {
    prev_state: prev.prev_state || null,
    prev_action: action_idx
  });
}
