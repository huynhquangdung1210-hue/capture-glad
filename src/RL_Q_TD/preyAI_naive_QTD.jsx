// ----------------------------------------------
// RL Prey AI System with starvation and rewards
// ----------------------------------------------

import { length, randRange } from "./utils.jsx";
import { rlSocket } from "./rlConnection.jsx";

export const AIType = {
  HEURISTICS: "heuristics",
  RL: "rl",
};
export const RL_WEIGHTS={specialeff:7, //per predator killed
                        specialwaste: -10, //per unused special
                        starve:-20,
                        foodgain:1.0, //per star eaten
                        survival:0.01,
                        reproduce:5.0 //per offspring
}

export const PREY_STATS={special_cost:5, //perenergy,
                        reproduce_cost:5,
                        specialkillradius:50,
                        specialtime:60,

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

/* -----------------------------
   Heuristic movement
/* ----------------------------
   Return the closest predator within dangerRadius.
   If none are close enough, return null.
------------------------------*/
function findNearbyPredator(prey, predators) {
  const dangerRadius = 50;

  let closest = null;
  let closestDist = Infinity; // track closest predator anywhere
  let predatorNearby = false;

  for (const p of predators) {
    const d = length(p.x - prey.x, p.y - prey.y);

    if (d < closestDist) {
      closestDist = d;
      closest = p;
    }

    if (d < dangerRadius) {
      predatorNearby = true;
    }
  }

  const closestCoords = closest ? { x: closest.x, y: closest.y } : null;

  return { predatorNearby, closestCoords };
}

function findNearestAndThird(prey, stars) {
  if (stars.length === 0) {
    return { nearest: null, third: null };
  }

  // Sort stars by distance ascending
  const sorted = [...stars].sort((a, b) => {
    const da = Math.hypot(a.x - prey.x, a.y - prey.y);
    const db = Math.hypot(b.x - prey.x, b.y - prey.y);
    return da - db;
  });
  const nearest = sorted[0] || null;
  const third   = sorted[2] || null;  // index 2 = 3rd star

  return { nearest, third };
}


function heuristicPrey(prey, stars, predators) {
  // --- Check predators nearby
  const { predatorNearby, closestCoords } = findNearbyPredator(prey, predators);
  let prednearby=predatorNearby;
  let closestPred=closestCoords;
  // --- Decide food action
  let food_action = "IDLE_FOOD";

  if (prednearby && prey.special == false && prey.energy >= PREY_STATS.special_cost) {
    food_action = "ACTIVATE_SPECIAL";
  } else if (prey.energy >= PREY_STATS.reproduce_cost) {
    food_action = "REPRODUCE";
  }
  let nearestfood,thirdfood = findNearestAndThird(prey, stars);
  if (!nearestfood) { 
    return { dx: randRange(-1, 1), dy: randRange(-1, 1), food_action, foodAction: food_action };
  }
  // --- Compute movement towards nearest food
  const dx = nearestfood.x - prey.x;
  const dy = nearestfood.y - prey.y;

  return { dx, dy, food_action, foodAction: food_action };
}
/* ---------------------------------------------
   STARVATION, FOOD, AND PREDATOR INTERACTIONS
----------------------------------------------*/
  
export function getRLState(prey, stars, predators) {
  const { predatorNearby, closestCoords } = findNearbyPredator(prey, predators);
  let prednearby=predatorNearby;
  let closestpred=closestCoords;
  if (closestCoords===null){
    closestpred={x:0,y:0};
  }
  
  let {nearest: nearestfood, third: thirdfood} = findNearestAndThird(prey, stars);
    if (nearestfood===null){
    nearestfood={x:0,y:0};
  }
  // ---------------------------
  // --- Compute movement towards nearest food
  const dx_food = nearestfood.x - prey.x;
  const dy_food = nearestfood.y - prey.y;
  const d_from_center = length(prey.x, prey.y);
  // ---------------------------
  // --- Compute movement towards away predator
  const dx_pred = -(closestpred.x - prey.x);
  const dy_pred = -(closestpred.y - prey.y);
  
  // ---------------------------
  // 4. Prey internal state
  // ---------------------------
  let starving = false;
  if (prey.energy <= 1) {
    starving = true;
  }
  return [
    // nearest food info
    dx_food,
    dy_food,
    // nearest predator info
    dx_pred,
    dy_pred,
    // binary flags
    prey.energy,
    prednearby,
    starving,
  ];
}

//* -----------------------------
  //  RL Prey Logic
// ------------------------------*/
function rlPrey(prey, stars, predators, getRLState) {
  // --------------------------
  // 1. Build RL state vector
  // --------------------------
  const state = getRLState(prey, stars, predators);
  prey.rlPrevState = state;
  console.log(prey.rlReward)
  const prev = rlPreyMap.get(prey.id) || {};
  // --------------------------
  // 2. Send RL transition to Python
  // --------------------------
  if (rlSocket.readyState === WebSocket.OPEN) {
    rlSocket.send(JSON.stringify({
      state,
      reward: prey.rlReward || 0,
      prev_state: prev.prev_state || null,
      prev_action: prev.prev_action || null,
      preyId: prey.id
    }));
  }

  // Reset reward after sending it
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

/* -------------------------------------
   Factory to create Prey AI controller
--------------------------------------*/
export function createPreyAI({ type = AIType.HEURISTICS, getRLState = null } = {}) {
  return {
    getAction(prey, stars, predators) {
      if (type === AIType.HEURISTICS) {
        return heuristicPrey(prey, stars, predators);
      }
      if (type === AIType.RL) {
        return rlPrey(prey, stars, predators, getRLState);
      }
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
