"""Simple WebSocket RL server for testing.

Provides a basic server that accepts state messages and returns
dummy actions for testing the WebSocket connection.
"""
import asyncio
import websockets
import json

async def handler(ws):
    print("Client connected")
    async for message in ws:
        data = json.loads(message)
        prey_id = data.get("preyId")
        state = data.get("state")
        # dummy action for testing
        action = {"dx": 0, "dy": 1, "action_idx": 0, "preyId": prey_id}
        await ws.send(json.dumps(action))

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Python RL server listening on ws://localhost:8765")
        await asyncio.Future()  # run forever

asyncio.run(main())
