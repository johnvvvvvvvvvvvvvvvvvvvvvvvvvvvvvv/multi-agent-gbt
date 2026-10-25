"""
Web server for the multi-agent system.

Setup:
    pip install fastapi uvicorn openai
    export OPENAI_API_KEY=your-key
    uvicorn server:app --reload
(Optional: export ACCESS_KEY=some-secret to require a key.)
Then open http://localhost:8000
"""
import asyncio
import json
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from multi_agent import Agent, orchestrator, parse_plan, synthesizer

app = FastAPI()
HERE = Path(__file__).parent
ACCESS_KEY = os.getenv("ACCESS_KEY")  # if set, /run requires ?key=<ACCESS_KEY>


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def pipeline(goal: str):
    try:
        yield sse("status", {"message": "Planning..."})
        plan = parse_plan(await orchestrator.run(goal))
        yield sse("plan", {"steps": plan})

        async def work(i: int, step: dict):
            worker = Agent(
                step["role"],
                f"You are an expert {step['role']}. Be concise and concrete.",
            )
            return i, await worker.run(step["task"])

        tasks = [asyncio.create_task(work(i, s)) for i, s in enumerate(plan)]
        results = {}
        for fut in asyncio.as_completed(tasks):  # emit each result as it finishes
            i, output = await fut
            results[i] = output
            yield sse("worker_done", {"index": i, "output": output})

        yield sse("status", {"message": "Synthesizing..."})
        combined = "\n\n".join(
            f"## {plan[i]['role']}\n{results[i]}" for i in sorted(results)
        )
        final = await synthesizer.run(f"Goal: {goal}\n\nSpecialist outputs:\n{combined}")
        yield sse("final", {"output": final})
    except Exception as e:
        yield sse("error", {"message": str(e)})


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/run")
async def run(goal: str, key: str = ""):
    if ACCESS_KEY and not secrets.compare_digest(key, ACCESS_KEY):
        raise HTTPException(status_code=401, detail="Invalid access key")
    return StreamingResponse(
        pipeline(goal),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
