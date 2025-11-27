
/* ----------------------------
   Internal Modules (encapsulated)
   ----------------------------*/
import {DEFAULTS} from "./Game.jsx"
import { length } from "./utils.jsx";
/* Physics module: entity creation, collisions, movement helpers */
export function createPhysics(arenaRadius) {
    function randomPos() {
        // uniform inside circle
        const r = Math.sqrt(Math.random()) * (arenaRadius * 0.8);
        const theta = Math.random() * Math.PI * 2;
        return { x: Math.cos(theta) * r, y: Math.sin(theta) * r };
    }

    function spawnPredator() {
        const p = randomPos();
        return {
        id: "pred-" + Math.random().toString(36).slice(2, 9),
        type: "predator",
        x: p.x,
        y: p.y,
        hunger: DEFAULTS.PREDATOR_HUNGER_SECS,
        eatenCount: 0,
        };
    }
    function reproducePredator(pred) {
    return {
        id: "pred-" + Math.random().toString(36).slice(2, 9),
        type: "predator",
        x: pred.x,
        y: pred.y,
        hunger: DEFAULTS.PREDATOR_HUNGER_SECS,
        eatenCount: 0,
    };
    }

    function spawnPrey() {
        const p = randomPos();
            // define angle + speed here
        const angle = Math.random() * Math.PI * 2;
        const speed = DEFAULTS.PREY_SPEED || 1;
        return {
        id: "prey-" + Math.random().toString(36).slice(2, 9),
        type: "prey",
        x: p.x,
        y: p.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        energy: 3,
        starvationTimer: 0,
        special: 0,
        alive: true,
        rlReward: 0,
        justeat:false,
        wandercooldown:0
        };
    }
    function reproducePrey(prey) {
            // define angle + speed here
        const angle = Math.random() * Math.PI * 2;
        const speed = DEFAULTS.PREY_SPEED || 1;
        return {
        id: "prey-" + Math.random().toString(36).slice(2, 9),
        type: "prey",
        x: prey.x,
        y: prey.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        energy: 3,
        starvationTimer: 0,
        special: false,
        alive: true,
        rlReward: 0,
        wandercooldown:0
        };
    }

    function spawnStar() {
        const p = randomPos();

        return {
        id: "star-" + Math.random().toString(36).slice(2, 9),
        type: "star",
        x: p.x,
        y: p.y,
        };
    }

    function keepInside(e) {
        const d = length(e.x, e.y);
        if (d>300){
            console.log(d, e.x, e.y)
        }
        if (d > arenaRadius) {
        e.x *= arenaRadius / d;
        e.y *= arenaRadius / d;
        }
    }

    function wrapAroundTorus(e) {
        // Toroidal wrapping: treat the square [-arenaRadius, arenaRadius] x [-arenaRadius, arenaRadius] as a torus
        // Entities wrapping on one side pop out on the other
        const bound = arenaRadius;
        if (e.x > bound) {
            e.x -= 2 * bound;
        } else if (e.x < -bound) {
            e.x += 2 * bound;
        }
        if (e.y > bound) {
            e.y -= 2 * bound;
        } else if (e.y < -bound) {
            e.y += 2 * bound;
        }
    }

  return {
    randomPos,
    spawnPredator,
    reproducePredator,
    spawnPrey,
    reproducePrey,
    spawnStar,
    keepInside,
    wrapAroundTorus,
  };
}