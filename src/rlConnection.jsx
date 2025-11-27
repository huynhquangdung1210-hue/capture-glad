export const rlSocket = new WebSocket("ws://localhost:9876");
import { registerRLAction } from "./preyAI.jsx"; // or wherever you define it

rlSocket.onopen = () => {
  console.log("RL WebSocket connected!");
};

rlSocket.onmessage = (msg) => {
  const data = JSON.parse(msg.data);
  const { preyId, dx, dy, action_idx, moveAction, foodAction } = data;
  registerRLAction(preyId, dx, dy, action_idx, moveAction, foodAction);
};
