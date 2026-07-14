"""
base_agent.py
Base class all agents inherit. Handles API calls, the agentic tool loop,
and token usage tracking.

MODEL TIERING (token cost optimization):
- claude-haiku-4-5   → schema mapping, award classification, risk scoring
                        (simple structured tasks — 12x cheaper than Sonnet)
- claude-sonnet-4-6  → compliance orchestration, report writing
                        (needs stronger reasoning / prose quality)
"""

import json
import anthropic

# Shared token ledger across all agents in a pipeline run
TOKEN_LEDGER = {"input": 0, "output": 0, "calls": 0}

# Approx pricing per million tokens (for the cost estimate printout)
PRICING = {
    "claude-haiku-4-5-20251001": {"in": 1.00,  "out": 5.00},
    "claude-sonnet-4-6":          {"in": 3.00,  "out": 15.00},
}


def reset_token_ledger():
    TOKEN_LEDGER.update({"input": 0, "output": 0, "calls": 0, "cost_usd": 0.0})


def get_token_summary() -> dict:
    return dict(TOKEN_LEDGER)


class BaseAgent:

    def __init__(self, client: anthropic.Anthropic, verbose: bool = True):
        self.client  = client
        self.model   = "claude-sonnet-4-6"   # default; agents override
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _track(self, response):
        """Record token usage from an API response."""
        usage = getattr(response, "usage", None)
        if usage:
            TOKEN_LEDGER["input"]  += usage.input_tokens
            TOKEN_LEDGER["output"] += usage.output_tokens
            TOKEN_LEDGER["calls"]  += 1
            price = PRICING.get(self.model, PRICING["claude-sonnet-4-6"])
            TOKEN_LEDGER["cost_usd"] = TOKEN_LEDGER.get("cost_usd", 0.0) + (
                usage.input_tokens  / 1_000_000 * price["in"] +
                usage.output_tokens / 1_000_000 * price["out"]
            )

    def call(self, system: str, user_message: str, max_tokens: int = 4096) -> str:
        """Single-turn call — no tools."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        self._track(response)
        return response.content[0].text

    def run_loop(
        self,
        system: str,
        user_message: str,
        tools: list,
        tool_executor,
        max_tokens: int = 4096,
    ) -> str:
        """Agentic loop with tool use."""
        messages = [{"role": "user", "content": user_message}]

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
            self._track(response)

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return next(
                    (block.text for block in response.content if hasattr(block, "text")),
                    "",
                )

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        self._log(f"      🔧 {block.name}")
                        result = tool_executor(block.name, block.input)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     json.dumps(result, default=str),
                        })
                messages.append({"role": "user", "content": tool_results})

            else:
                break

        return "Agent did not complete."
