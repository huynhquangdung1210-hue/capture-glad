/* Game.jsx placeholder. Please paste from canvas manually. */import React, { useEffect, useRef, useState } from "react";
import { createPreyAI, AIType,getRLState, RL_WEIGHTS,PREY_STATS } from "./preyAI.jsx";
import { randRange, clamp, length, getRelativePolar,insideArena} from "./utils.jsx"
import {createPhysics} from "./physicsspawn.jsx"
/*
  Capture-the-Glad Game (single-file Game.jsx)
  - Predator: square
  - Prey: circle (collect stars to survive; power star kills predator)
  - Stars: regular and power
  - Arena: circle
  - Reproduction doubling per configurable interval
  - Predator must eat 1 prey per second (hunger timer)
  - Features: UI controls, HUD, replay (record/play/export/import), modular internal structure
*/

/* ----------------------------
   Configuration Defaults
   ----------------------------*/
export const DEFAULTS = {
  CANVAS_SIZE: 700,
  ARENA_RADIUS: 300,
  INITIAL_PREDATORS: 2,
  INITIAL_PREY: 6,
  STAR_COUNT: 15,
  SPECIAL_PREY_SPAWN_PER_SEC: 0.5,
  PREDATOR_SPEED: 1.8,
  PREY_SPEED: 2.2,
  // Reproduce after eating this many food items
  PREDATOR_REPRO_EAT: 5, // predators reproduce after eating 5 prey
  FIXED_TIMESTEP: 1 / 60, // 60 ticks per second logical update
  PREDATOR_HUNGER_SECS: 5, // must eat 1 prey per second (resets to this value when eat)
  MAX_EPISODE_TIME: 300, // maximum duration of an episode in seconds
  PREDATOR_SPAWN_INTERVAL: 10, // seconds between predator spawns
  PREDATOR_SPAWN_RAMP: 0.95,   // reduce interval by 5% each spawn (optional)
};


/* AI module: simple heuristics, also provides a hook where models can plug in */
function createAI() {
  function predatorAction(pred, preyList) {
    // chase nearest prey
    if (preyList.length === 0) {
      // wander
      return { dx: randRange(-1, 1), dy: randRange(-1, 1) };
    }
    let nearest = preyList[0];
    let best = Infinity;
    for (const p of preyList) {
      const d = length(p.x - pred.x, p.y - pred.y);
      if (d < best) {
        best = d;
        nearest = p;
      }
    }
    return { dx: nearest.x - pred.x, dy: nearest.y - pred.y };
  }


  return {
    predatorAction,
  };

}

/* Renderer: draws everything on a canvas 2D context */
function createRenderer(ctx, canvasSize, arenaRadius) {
  const center = canvasSize / 2;
  function clear() {
    ctx.clearRect(0, 0, canvasSize, canvasSize);
  }
  function drawArena() {
    ctx.save();
    ctx.translate(center, center);
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(0, 0, arenaRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
  function worldToCanvas(x, y) {
    return { cx: Math.round(center + x), cy: Math.round(center + y) };
  }
  function drawStar(star) {
    const { cx, cy } = worldToCanvas(star.x, star.y);
    ctx.beginPath();
    ctx.fillStyle = "#e6c200";
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  function drawPrey(pr) {
      // --- Draw prey as a small velocity-oriented arrow ---
      const { cx, cy } = worldToCanvas(pr.x, pr.y);

      const vx = Number.isFinite(pr.vx) ? pr.vx : 0.0001;
      const vy = Number.isFinite(pr.vy) ? pr.vy : 0.0001;

      let angle = Math.atan2(vy, vx);
      if (!Number.isFinite(angle)) angle = 0;

      // size (adjustable)
      const size = pr.special ? 10 : 7;
      const half = size * 0.5;

      // Arrow coordinates
      const tipX   = cx + Math.cos(angle) * size;
      const tipY   = cy + Math.sin(angle) * size;

      const leftX  = cx + Math.cos(angle + Math.PI * 0.75) * half;
      const leftY  = cy + Math.sin(angle + Math.PI * 0.75) * half;

      const rightX = cx + Math.cos(angle - Math.PI * 0.75) * half;
      const rightY = cy + Math.sin(angle - Math.PI * 0.75) * half;

      // Draw arrow
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(leftX, leftY);
      ctx.lineTo(rightX, rightY);
      ctx.closePath();
      const LOW_ENERGY_THRESHOLD = 0; // configurable threshold
      ctx.fillStyle = pr.energy <= LOW_ENERGY_THRESHOLD ? "#dc771eff" : "#1976d2";
      ctx.fill();

    // Indicator: if prey just activated special this tick, draw a red circle size of the radius of effect
    if (pr.special) {
      const specialKillRadius = PREY_STATS.specialkillradius;
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = "red";
      ctx.lineWidth = 3;
      ctx.arc(cx, cy, specialKillRadius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }


  }
  function drawPredator(pred) {
    const { cx, cy } = worldToCanvas(pred.x, pred.y);
    ctx.save();
    ctx.fillStyle = "#111";
    ctx.fillRect(cx - 8, cy - 8, 16, 16);
    ctx.restore();
  }
  function drawHUDText(text, x, y) {
    ctx.save();
    ctx.font = "14px monospace";
    ctx.fillStyle = "#222";
    ctx.fillText(text, x, y);
    ctx.restore();
  }
  return {
    clear,
    drawArena,
    drawStar,
    drawPrey,
    drawPredator,
    drawHUDText,
    worldToCanvas,
  };
}

/* ----------------------------
   Main React Component
   ----------------------------*/
export default function Game() {
  // canvas refs & constants
  const canvasRef = useRef(null);
  const [canvasSize] = useState(DEFAULTS.CANVAS_SIZE);
  const [arenaRadius, setArenaRadius] = useState(DEFAULTS.ARENA_RADIUS);

  // physics, ai, renderer refs (we'll initialize after mount)
  const physicsRef = useRef(null);
  const aiRef = useRef(null);
  const rendererRef = useRef(null);

  // Entities in state (we keep in refs for fast updates, and mirror counts in state for UI)
  const predatorsRef = useRef([]);
  const preyRef = useRef([]);
  const starsRef = useRef([]);

  // UI state / control parameters
  const [predatorSpeed, setPredatorSpeed] = useState(DEFAULTS.PREDATOR_SPEED);
  const [preySpeed, setPreySpeed] = useState(DEFAULTS.PREY_SPEED);
  const [starCount, setStarCount] = useState(DEFAULTS.STAR_COUNT);
  // Removed power stars: spawn special prey instead
  const [predatorReproCount, setPredatorReproCount] = useState(DEFAULTS.PREDATOR_REPRO_EAT);
  const [preyReproCount, setPreyReproCount] = useState(DEFAULTS.PREY_REPRO_EAT);
  const [running, setRunning] = useState(true);
  const [showVectors, setShowVectors] = useState(false);

  const [tickCount, setTickCount] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef(0);
  const fpsRef = useRef(0);

  // Replay & recording
  const recordingRef = useRef(false);
  const replayBufferRef = useRef([]); // array of snapshots
  const [isPlayingReplay, setIsPlayingReplay] = useState(false);
  const replayIndexRef = useRef(0);

  // Training hooks: actions can be passed as a map id->action (dx,dy)
  const externalActionsRef = useRef({}); // {entityId: {dx,dy}}

  // Misc refs
  const lastTimeRef = useRef(null);
  const accumulatorRef = useRef(0);
  const fpsSmootherRef = useRef({ lastFpsTime: performance.now(), frames: 0 });
  //Predator Spawn
  let lastPredatorSpawnTime = 0; // timestamp of last spawn
  let predatorSpawnInterval = DEFAULTS.PREDATOR_SPAWN_INTERVAL; // seconds
  let predatorSpawnRamp = DEFAULTS.PREDATOR_SPAWN_RAMP;   // reduce interval by 5% each spawn (optional)

  // initialize on mount
  useEffect(() => {
    // init modules
    physicsRef.current = createPhysics(arenaRadius);
    aiRef.current = createAI();

    // populate entities
    predatorsRef.current = Array.from({ length: DEFAULTS.INITIAL_PREDATORS }, () =>
      physicsRef.current.spawnPredator()
    );
    preyRef.current = Array.from({ length: DEFAULTS.INITIAL_PREY }, () => physicsRef.current.spawnPrey());
    starsRef.current = Array.from({ length: DEFAULTS.STAR_COUNT }, () => physicsRef.current.spawnStar());

    // renderer: wait for canvas
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    rendererRef.current = createRenderer(ctx, canvasSize, arenaRadius);
  }, []); // run once


  /* ----------------------------
     Game Logic: step function
     - This is the authoritative simulation "step" that applies dt seconds of simulation.
     - Externally usable via step(actionMap) for training or headless control.
  ----------------------------*/

  function step(dt) {
    const physics = physicsRef.current;
    const ai = aiRef.current;
    if (!ai) return; // <-- early exit if AI not ready

    // Simple actions map: externalActionsRef contains user-provided actions
    const actions = externalActionsRef.current || {};

    // --- Move predators according to AI or external actions
    const predators = predatorsRef.current;
    const preys = preyRef.current;
    const stars = starsRef.current;
    // Example: define a simple state discretization function
    

    for (const pred of predators) {
      let act = actions[pred.id];
      if (act) {
        const mag = Math.hypot(act.dx, act.dy) || 1;
        pred.x += (act.dx / mag) * predatorSpeed * dt * 60;
        pred.y += (act.dy / mag) * predatorSpeed * dt * 60;
      } else {
        const a = ai.predatorAction(pred, preys);
        const mag = Math.hypot(a.dx, a.dy) || 1;
        pred.x += (a.dx / mag) * predatorSpeed * dt * 60;
        pred.y += (a.dy / mag) * predatorSpeed * dt * 60;
      }
      physics.wrapAroundTorus(pred);
    }
    const preyAI = createPreyAI({ type: AIType.RL, getRLState });

    // --- Move preys
    for (let j = preys.length - 1; j >= 0; j--) {
      const pr = preys[j];      let act = actions[pr.id];
      if (act) {
        const mag = Math.hypot(act.dx, act.dy) || 1;
        pr.vx = (act.dx / mag) * preySpeed;
        pr.vy = (act.dy / mag) * preySpeed;
        pr.x += (act.dx / mag) * preySpeed * dt * 60;
        pr.y += (act.dy / mag) * preySpeed * dt * 60;
      } else {
        pr.special-=1*dt*60;
        pr.rlReward += RL_WEIGHTS.survival*dt*60; // small reward for surviving each tick
        if (pr.special < 0) pr.special = 0;
        if (pr.alive===false) pr.rlReward += RL_WEIGHTS.starve; // prey punished for being eaten

        const a = preyAI.getAction(pr, stars, predators);
        const mag = Math.hypot(a.dx, a.dy) || 1;
        if (pr.alive === false) preys.splice(j, 1);
        pr.vx = (a.dx / mag) * preySpeed;
        pr.vy = (a.dy / mag) * preySpeed;
        pr.x += (a.dx / mag) * preySpeed * dt * 60;
        pr.y += (a.dy / mag) * preySpeed * dt * 60;
        const foodAction = a.foodAction || a.food_action || a.foodAction;
        if (foodAction === "ACTIVATE_SPECIAL") {
          if ((pr.energy || 0) >= PREY_STATS.special_cost) {
            pr.special = PREY_STATS.specialtime; // active for 3 seconds
            pr.energy -= PREY_STATS.special_cost;
          }
        } else if (foodAction === "REPRODUCE") {
          if ((pr.energy || 0) >= PREY_STATS.reproduce_cost) {
            pr.energy -= PREY_STATS.reproduce_cost;
            const baby = physics.reproducePrey(pr);
            preyRef.current.push(baby);
          }
        }
      }
      physics.wrapAroundTorus(pr);
    }
    // --- Prey activated abilities: special attack or reproduction (energy-based)
    const predEatRange = 12;
    const predatorsToRemove = new Set();
    const specialKillRadius = PREY_STATS.specialkillradius; // radius within which prey special kills nearby predators

    // Prey may choose to activate special (cost 4 energy) if predators are nearby,
    // or reproduce by spending `preyReproCount` energy. This must happen before
    // predator-eat collision resolution so special kills take effect immediately.
    for (const pr of preyRef.current) {
      if (pr.special > 0) {
        predatorsRef.current.forEach(pred => {
          if (length(pred.x - pr.x, pred.y - pr.y) <= specialKillRadius) {
            predatorsToRemove.add(pred.id);
          }
        });
        pr.rlReward += RL_WEIGHTS.specialeff * predatorsToRemove.size;
        if (predatorsToRemove.size === 0) {
          pr.rlReward += RL_WEIGHTS.specialwaste;
        }
      }
    }


      // --- Collisions: predators eat prey
    for (let i = predatorsRef.current.length - 1; i >= 0; i--) {
      const P = predatorsRef.current[i];
      for (let j = preyRef.current.length - 1; j >= 0; j--) {
        const pr = preyRef.current[j];
          if (length(P.x - pr.x, P.y - pr.y) < predEatRange) {
            const wasSpecial = !!pr.special;
            // Remove the prey
            pr.alive=false;
            // --- Normal pred eats prey ---
            P.hunger = DEFAULTS.PREDATOR_HUNGER_SECS;
            P.eatenCount = (P.eatenCount || 0) + 1;
            // Reproduce if eaten enough
            if (P.eatenCount >= predatorReproCount) {
              predatorsRef.current.push(physics.reproducePredator(P));
              P.eatenCount = 0;

            }
          break; // move to next predator
          }

      }
    }

    // --- Remove predators killed by prey special safely ---
    if (predatorsToRemove.size > 0) {
      predatorsRef.current = predatorsRef.current.filter(p => !predatorsToRemove.has(p.id));
    }
    // --- Prey starvation & energy update
    for (let i = preyRef.current.length - 1; i >= 0; i--) {
      const pr = preyRef.current[i];

      // Increment starvation timer by dt
      pr.starvationTimer = (pr.starvationTimer || 0) + dt;

      // Energy decay every 5 seconds
      if (pr.starvationTimer >= 5) {
        pr.energy = (pr.energy || 0) - 1;
        pr.starvationTimer = 0;
      }

      // Remove prey if energy is below 0
  
      if ((pr.energy || 0) <= -1) {
        pr.alive=false;
        continue;
      }

      // Reset justEat flag; it will be set true if prey eats a star this step
      pr.justeat = false;
    }

    // --- Prey eating stars
    for (const pr of preyRef.current) {
      for (let j = starsRef.current.length - 1; j >= 0; j--) {
        const s = starsRef.current[j];
        if (length(pr.x - s.x, pr.y - s.y) < 10) {
          const prevEnergy = Number.isFinite(pr.energy) ? pr.energy : Number(pr.energy) || 0;
          pr.energy = prevEnergy + 1;
          pr.justeat = true;
          pr.rlReward = pr.rlReward + RL_WEIGHTS.foodgain;
          starsRef.current.splice(j, 1);
          break; // only eat one star per step
        }
      }
    }
    // --- Predator starvation
    for (let i = predatorsRef.current.length - 1; i >= 0; i--) {
      const P = predatorsRef.current[i];
      P.hunger -= dt;
      if (P.hunger <= 0) predatorsRef.current.splice(i, 1);
    }

    // --- Respawn stars gradually to meet target counts
    while (starsRef.current.length < starCount) {
      starsRef.current.push(physics.spawnStar());
      if (starsRef.current.length >= starCount) break;
    }
      // Check if episode is done
    if (preyRef.current.length === 0 || elapsedRef.current >= DEFAULTS.MAX_EPISODE_TIME) {

      resetWorld(); // start new episode
    }
    const currentTime = performance.now() / 1000; // seconds
    
    if (currentTime - lastPredatorSpawnTime >= predatorSpawnInterval) {
      // spawn one predator
      predators.push(physics.spawnPredator());
      lastPredatorSpawnTime = currentTime;

      // optionally ramp up spawn rate
      predatorSpawnInterval = Math.max(1, predatorSpawnInterval * predatorSpawnRamp);
    }

    // update HUD counters occasionally
    setTickCount((t) => t + 1);

    // record to replay buffer if recording
    if (recordingRef.current) {
      replayBufferRef.current.push({
        time: elapsedRef.current,
        predators: predatorsRef.current.map((p) => ({ id: p.id, x: p.x, y: p.y })),
        prey: preyRef.current.map((p) => ({ id: p.id, x: p.x, y: p.y, special: p.special })),
        stars: starsRef.current.map((s) => ({ id: s.id, x: s.x, y: s.y, type: "star" })),
      });
      if (replayBufferRef.current.length > 20000) replayBufferRef.current.shift();
    }
  }

  /* ----------------------------
     Game Loop (fixed-step integration)
     ----------------------------*/
  useEffect(() => {
    let raf = null;
    lastTimeRef.current = performance.now();
    accumulatorRef.current = 0;

    function loop(now) {
      // compute dt in seconds
      let frameDt = (now - lastTimeRef.current) / 1000;
      if (frameDt > 0.25) frameDt = 0.25; // clamp huge frame time
      lastTimeRef.current = now;
      accumulatorRef.current += frameDt;

      // update FPS
      fpsSmootherRef.current.frames++;
      const fpsNow = performance.now();
      if (fpsNow - fpsSmootherRef.current.lastFpsTime >= 1000) {
        fpsRef.current = fpsSmootherRef.current.frames;
        fpsSmootherRef.current.frames = 0;
        fpsSmootherRef.current.lastFpsTime = fpsNow;
      }

      // fixed timestep
      const ts = DEFAULTS.FIXED_TIMESTEP;
      while (accumulatorRef.current >= ts) {
        if (!running) break;
        step(ts);
        accumulatorRef.current -= ts;
        elapsedRef.current += ts;
      }

      // render
      const renderer = rendererRef.current;
      if (renderer) {
        const ctx = canvasRef.current.getContext("2d");
        renderer.clear();
        // renderer.drawArena(); // Hide arena circle for toroidal wrapping

        // draw stars
        for (const s of starsRef.current) renderer.drawStar(s);

        // draw preys
        for (const p of preyRef.current) renderer.drawPrey(p);

        // draw predators
        for (const pd of predatorsRef.current) renderer.drawPredator(pd);

        // HUD overlay
        const hudX = 12;
        const hudYStart = 20;
        renderer.drawHUDText(`Predators: ${predatorsRef.current.length}`, hudX, hudYStart);
        renderer.drawHUDText(`Prey: ${preyRef.current.length}`, hudX, hudYStart + 18);
        renderer.drawHUDText(`Stars: ${starsRef.current.length}`, hudX, hudYStart + 36);
        const specialCount = preyRef.current.filter((p) => p.special).length;
        renderer.drawHUDText(`Special prey: ${specialCount}`, hudX, hudYStart + 54);
        renderer.drawHUDText(`Elapsed: ${elapsedRef.current.toFixed(1)}s`, hudX, hudYStart + 72);
        renderer.drawHUDText(`FPS: ${fpsRef.current}`, hudX, hudYStart + 90);
        renderer.drawHUDText(`Recording: ${recordingRef.current ? "YES" : "NO"}`, hudX, hudYStart + 108);
        renderer.drawHUDText(`Legend:`, hudX, hudYStart + 126);
        renderer.drawHUDText(`Red dot = Low energy (<=1)`, hudX, hudYStart + 144);
        renderer.drawHUDText(`Orange ring = Just activated special`, hudX, hudYStart + 162);
      }

      // update React HUD state occasionally
      setElapsed(elapsedRef.current);

      if (!isPlayingReplay) {
        raf = requestAnimationFrame(loop);
      } else {
        // when playing replay, render frames from buffer
        raf = requestAnimationFrame(loop);
      }
    }

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, predatorSpeed, preySpeed, starCount, arenaRadius, predatorReproCount, preyReproCount]);

  /* ----------------------------
     Controls & UI actions
  ----------------------------*/
  function resetWorld() {
    const physics = physicsRef.current;
    predatorsRef.current = Array.from({ length: DEFAULTS.INITIAL_PREDATORS }, () =>
      physics.spawnPredator()
    );
    preyRef.current = Array.from({ length: DEFAULTS.INITIAL_PREY }, () => physics.spawnPrey());
    starsRef.current = Array.from({ length: starCount }, () => physics.spawnStar());
    replayBufferRef.current = [];
    elapsedRef.current = 0;
    setTickCount(0);
  }

  function toggleRecording() {
    recordingRef.current = !recordingRef.current;
  }

  function exportReplay() {
    const data = JSON.stringify(replayBufferRef.current);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `replay_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importReplay(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const arr = JSON.parse(e.target.result);
        replayBufferRef.current = arr;
        alert("Replay loaded: " + arr.length + " frames");
      } catch (err) {
        alert("Invalid replay file");
      }
    };
    reader.readAsText(file);
  }

  function playReplay() {
    if (replayBufferRef.current.length === 0) {
      alert("No replay recorded or loaded.");
      return;
    }
    // simple playback: freeze simulation and iterate frames
    setRunning(false);
    setIsPlayingReplay(true);
    let idx = 0;
    function stepPlay() {
      if (idx >= replayBufferRef.current.length) {
        setIsPlayingReplay(false);
        setRunning(true);
        return;
      }
      const frame = replayBufferRef.current[idx];
      // set entities positions for rendering only
      // (we won't affect simulation refs permanently; replace current arrays for visualization)
      predatorsRef.current = frame.predators.map((p) => ({ id: p.id, x: p.x, y: p.y }));
      preyRef.current = frame.prey.map((p) => ({ id: p.id, x: p.x, y: p.y, special: p.special }));
      starsRef.current = frame.stars.map((s) => ({ id: s.id, x: s.x, y: s.y }));
      idx++;
      replayIndexRef.current = idx;
      setTimeout(stepPlay, 1000 / 30); // play 30 fps
    }
    stepPlay();
  }

  function logAllPrey() {
    try {
      const arr = preyRef.current.map((p) => ({
        id: p.id,
        x: Number(p.x.toFixed(2)),
        y: Number(p.y.toFixed(2)),
        vx: Number((p.vx || 0).toFixed(2)),
        vy: Number((p.vy || 0).toFixed(2)),
        energy: p.energy || 0,
        special: p.special || 0,
        starvationTimer: p.starvationTimer || 0,
        justeat: !!p.justeat,
        alive: p.alive !== false,
        rlReward: p.rlReward || 0,
      }));
      console.group(`Prey snapshot (${arr.length})`);
      console.table(arr);
      console.groupEnd();
    } catch (err) {
      console.error('Failed to log prey info', err);
    }
  }

  /* ----------------------------
     Helper: spawn N entities
  ----------------------------*/
  function spawnN(type, n) {
    const p = physicsRef.current;
    if (type === "pred") {
      for (let i = 0; i < n; i++) predatorsRef.current.push(p.spawnPredator());
    } else if (type === "prey") {
      for (let i = 0; i < n; i++) preyRef.current.push(p.spawnPrey());
    } else if (type === "star") {
      for (let i = 0; i < n; i++) starsRef.current.push(p.spawnStar());
    }
  }

  /* ----------------------------
     Public API / training hook
     - stepWithActions(actionMap) allows external controllers to step simulation for dt seconds
  ----------------------------*/
  function stepWithActions(actionMap, dt = DEFAULTS.FIXED_TIMESTEP) {
    externalActionsRef.current = actionMap;
    step(dt);
    externalActionsRef.current = {};
  }

  /* ----------------------------
     JSX UI
  ----------------------------*/
  return (
    <div style={{ display: "flex", gap: 20, padding: 12, fontFamily: "system-ui, Arial" }}>
      <div>
        <canvas
          ref={canvasRef}
          width={canvasSize}
          height={canvasSize}
          style={{ border: "1px solid #ddd", background: "#fafafa" }}
        />
        <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
          <button onClick={() => setRunning((r) => !r)}>{running ? "Pause" : "Resume"}</button>
          <button
            onClick={() => {
              resetWorld();
            }}
          >
            Reset
          </button>
          <button
            onClick={() => {
              toggleRecording();
            }}
          >
            {recordingRef.current ? "Stop Rec" : "Start Rec"}
          </button>
          <button onClick={() => exportReplay()}>Export Replay</button>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            Load:
            <input
              type="file"
              accept="application/json"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) importReplay(e.target.files[0]);
              }}
            />
          </label>
          <button onClick={() => playReplay()}>Play Replay</button>
        </div>
      </div>

      {/* Controls panel */}
      <div style={{ minWidth: 340 }}>
        <h3>Controls</h3>

        <label>
          Arena radius: {arenaRadius}
          <input
            type="range"
            min="150"
            max="360"
            value={arenaRadius}
            onChange={(e) => {
              const v = parseInt(e.target.value);
              setArenaRadius(v);
              // re-initialize physics with new radius for spawn/clamping
              physicsRef.current = createPhysics(v);
            }}
          />
        </label>

        <hr />

        <label>
          Predator speed: {predatorSpeed.toFixed(2)}
          <input
            type="range"
            min="0.2"
            max="4.5"
            step="0.05"
            value={predatorSpeed}
            onChange={(e) => setPredatorSpeed(parseFloat(e.target.value))}
          />
        </label>
        <label>
          Prey speed: {preySpeed.toFixed(2)}
          <input
            type="range"
            min="0.2"
            max="6"
            step="0.05"
            value={preySpeed}
            onChange={(e) => setPreySpeed(parseFloat(e.target.value))}
          />
        </label>

        <hr />

        <label>
          Star target count: {starCount}
          <input
            type="range"
            min="0"
            max="200"
            value={starCount}
            onChange={(e) => setStarCount(parseInt(e.target.value))}
          />
        </label>
        <hr />

        <label>
          Predator reproduce after eating: {predatorReproCount}
          <input
            type="range"
            min="1"
            max="6"
            step="1"
            value={predatorReproCount}
            onChange={(e) => setPredatorReproCount(parseInt(e.target.value))}
          />
        </label>

        <label>
          Prey reproduce after eating: {preyReproCount}
          <input
            type="range"
            min="1"
            max="12"
            step="1"
            value={preyReproCount}
            onChange={(e) => setPreyReproCount(parseInt(e.target.value))}
          />
        </label>

        <hr />

        <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
          <button onClick={() => spawnN("pred", 1)}>Spawn Predator</button>
          <button onClick={() => spawnN("prey", 1)}>Spawn Prey</button>
          <button onClick={() => spawnN("star", 5)}>Spawn 5 Stars</button>
          <button onClick={() => spawnN("special", 1)}>Spawn Special</button>
          <button onClick={() => logAllPrey()}>Log Prey Info</button>
        </div>

        <div>
          <label>
            <input
              type="checkbox"
              checked={showVectors}
              onChange={(e) => setShowVectors(e.target.checked)}
            />{" "}
            Show debug vectors (not implemented: placeholder)
          </label>
        </div>

        <hr />

        {/* HUD stats */}
        <h4>HUD</h4>
        <div>Predators: {predatorsRef.current.length}</div>
        <div>Prey: {preyRef.current.length}</div>
        <div>Stars: {starsRef.current.length}</div>
        <div>Special prey: {preyRef.current.filter((p) => p.special).length}</div>
        <div>Elapsed (s): {elapsed.toFixed(1)}</div>
        <div>Ticks: {tickCount}</div>
      </div>
    </div>
  );
}

