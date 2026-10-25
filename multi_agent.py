"""
Minimal multi-agent system on the OpenAI API:
orchestrator -> parallel workers -> synthesizer.

Setup:
    pip install openai
    export OPENAI_API_KEY=your-key
    export OPENAI_MODEL=gpt-5-mini     # optional, this is the default
    python multi_agent.py "Research the pros and cons of remote work"
"""
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI

client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # change via env var, no code edit needed


class Agent:
    """An agent is just a name + a system prompt + an LLM call."""

    def __init__(self, name: str, system: str, json_mode: bool = False):
        self.name = name
        self.system = system
        self.json_mode = json_mode

    async def run(self, task: str) -> str:
        kwargs = {}
        if self.json_mode:
            # Forces valid JSON output (the prompt must also mention JSON).
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": task},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


orchestrator = Agent(
    "orchestrator",
    "You break a goal into 2-4 independent subtasks for specialist agents. "
    'Reply with ONLY a JSON object of this shape: '
    '{"subtasks": [{"role": "<specialist role>", "task": "<clear instruction>"}]}',
    json_mode=True,
)

synthesizer = Agent(
    "synthesizer",
    "You merge the outputs of several specialists into one coherent, "
    "well-structured final answer. Resolve contradictions and remove duplication.",
)


def parse_plan(text: str) -> list[dict]:
    data = json.loads(text)
    steps = [
        s for s in data.get("subtasks", [])
        if isinstance(s, dict) and s.get("role") and s.get("task")
    ]
    if not steps:
        raise ValueError(f"Orchestrator returned no usable subtasks:\n{text}")
    return steps[:4]


async def run_worker(role: str, task: str) -> tuple[str, str]:
    worker = Agent(role, f"You are an expert {role}. Be concise and concrete.")
    result = await worker.run(task)
    print(f"  [done] {role}")
    return role, result


async def main(goal: str) -> None:
    print("Planning...")
    plan = parse_plan(await orchestrator.run(goal))
    for step in plan:
        print(f"  - {step['role']}: {step['task']}")

    print("Running workers in parallel...")
    results = await asyncio.gather(
        *(run_worker(step["role"], step["task"]) for step in plan)
    )

    print("Synthesizing...")
    combined = "\n\n".join(f"## {role}\n{output}" for role, output in results)
    final = await synthesizer.run(f"Goal: {goal}\n\nSpecialist outputs:\n{combined}")

    print("\n=== FINAL ANSWER ===\n")
    print(final)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Plan a launch for a new productivity app"))
