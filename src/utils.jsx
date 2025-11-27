/**
 * Utility functions for game calculations.
 * Provides math helpers for distances, angles, and boundaries.
 */

/**
 * Returns a random number in [a, b).
 */
export const randRange = (a, b) => a + Math.random() * (b - a);

/**
 * Clamp a value between a minimum and maximum
 */
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/**
 * Euclidean distance / vector length
 */
export const length = (dx, dy) => Math.hypot(dx, dy);

/**
 * Check if a point is inside a circle
 */
export const insideArena = (x, y, radius) => length(x, y) <= radius;

// Utility to compute angle and distance
export function getRelativePolar(prey, target) {
  const dx = target.x - prey.x;
  const dy = target.y - prey.y;

  const dist = Math.sqrt(dx*dx + dy*dy);
  const angle = Math.atan2(dy, dx);  // radians (-π to π)

  return { dist, angle };
}
