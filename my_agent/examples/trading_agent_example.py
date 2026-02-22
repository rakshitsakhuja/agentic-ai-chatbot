"""
Example: Trading Agent
──────────────────────
Shows how to extend the boilerplate for a trading use case.
This wires up:
  - Custom trading tools (price fetch, indicators, order placement)
  - A risk-aware system prompt
  - Orchestrator with data / analysis / execution sub-agents
  - Planner for multi-step trading workflows
  - Reflection for decision quality checks

NOTHING here is production — no real exchange connections.
Replace the stub functions with your real APIs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from tools import ToolRegistry, register_builtin_tools
from memory import ShortTermMemory, KVStore, EpisodicStore
from agent import ReActAgent, Orchestrator, Planner, PlanExecutor, Reflector


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRADING TOOLS  — replace stubs with real exchange/data API calls
# ═══════════════════════════════════════════════════════════════════════════════

def register_trading_tools(registry: ToolRegistry):

    @registry.tool(
        description="Fetch the current spot price of a cryptocurrency.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "e.g. BTC, ETH, SOL"},
                "quote": {"type": "string", "description": "Quote currency (default USDT)", "default": "USDT"},
            },
            "required": ["symbol"],
        },
        tags=["trading", "data"],
    )
    def get_spot_price(symbol: str, quote: str = "USDT") -> str:
        # TODO: replace with real exchange API call
        # e.g. ccxt, binance-connector, pybit
        return f"STUB: {symbol}/{quote} price = 42000.00"

    @registry.tool(
        description="Fetch OHLCV candlestick data for a symbol.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "interval": {"type": "string", "description": "e.g. 1m, 5m, 1h, 1d", "default": "1h"},
                "limit": {"type": "integer", "description": "Number of candles", "default": 100},
            },
            "required": ["symbol"],
        },
        tags=["trading", "data"],
    )
    def get_ohlcv(symbol: str, interval: str = "1h", limit: int = 100) -> str:
        return f"STUB: {limit} candles of {symbol} @ {interval} — replace with real data"

    @registry.tool(
        description="Compute technical indicators (RSI, MACD, Bollinger Bands) for a symbol.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicators to compute e.g. ['RSI', 'MACD']",
                },
                "interval": {"type": "string", "default": "1h"},
            },
            "required": ["symbol", "indicators"],
        },
        tags=["trading", "analysis"],
    )
    def compute_indicators(symbol: str, indicators: list, interval: str = "1h") -> str:
        # TODO: use pandas-ta or talib
        return f"STUB: {symbol} indicators {indicators} @ {interval}"

    @registry.tool(
        description="Get the current funding rate for a perpetual futures contract.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
            },
            "required": ["symbol"],
        },
        tags=["trading", "data"],
    )
    def get_funding_rate(symbol: str) -> str:
        return f"STUB: {symbol} funding rate = 0.01% (8h)"

    @registry.tool(
        description="Get on-chain metrics: active addresses, NVT, exchange flows, etc.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "metric": {"type": "string", "description": "e.g. active_addresses, nvt_ratio, exchange_inflow"},
            },
            "required": ["symbol", "metric"],
        },
        tags=["trading", "onchain"],
    )
    def get_onchain_metric(symbol: str, metric: str) -> str:
        return f"STUB: {symbol} {metric} = 12345"

    @registry.tool(
        description="Check current open positions and portfolio balance.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        tags=["trading", "execution"],
    )
    def get_portfolio() -> str:
        return "STUB: BTC long 0.5 @ 41500 | USDT balance: 10000"

    @registry.tool(
        description=(
            "Place a trading order. IMPORTANT: validate against risk rules before calling. "
            "Returns order ID or error."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "order_type": {"type": "string", "enum": ["market", "limit", "stop"], "default": "market"},
                "quantity": {"type": "number", "description": "Position size in base currency"},
                "price": {"type": "number", "description": "Limit/stop price (optional for market)"},
                "reason": {"type": "string", "description": "Why this trade is being placed (for audit log)"},
            },
            "required": ["symbol", "side", "quantity", "reason"],
        },
        tags=["trading", "execution"],
    )
    def place_order(symbol: str, side: str, quantity: float, reason: str,
                    order_type: str = "market", price: float = None) -> str:
        # TODO: connect to exchange API, enforce risk checks
        return f"STUB: order placed — {side} {quantity} {symbol} @ {order_type} | reason: {reason}"

    @registry.tool(
        description="Check a trade against risk rules. Returns 'approved' or 'rejected: <reason>'.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string"},
                "quantity": {"type": "number"},
                "notional_usd": {"type": "number", "description": "USD value of the trade"},
            },
            "required": ["symbol", "side", "quantity", "notional_usd"],
        },
        tags=["trading", "risk"],
    )
    def check_risk(symbol: str, side: str, quantity: float, notional_usd: float) -> str:
        # TODO: implement real risk rules — max position, max drawdown, exposure limits
        MAX_NOTIONAL = 50_000
        if notional_usd > MAX_NOTIONAL:
            return f"rejected: notional ${notional_usd} exceeds limit ${MAX_NOTIONAL}"
        return "approved"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AGENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_SYSTEM_PROMPT = """You are an autonomous trading agent operating across crypto spot, perps, and prediction markets.

Core rules (NEVER violate):
1. ALWAYS call check_risk before place_order — never skip this step
2. NEVER size a position beyond the risk-approved notional
3. When uncertain about market direction, do NOT trade — wait or ask
4. Log every decision rationale using memory_store tool
5. If you hit an error, stop and report — do not retry blindly

Your workflow for each decision:
  a) Gather data (price, indicators, on-chain metrics, funding)
  b) Analyse market structure and regime
  c) Form a hypothesis with confidence level
  d) If confidence > 7/10 → size position → check risk → execute
  e) Record the decision and reasoning in memory"""


def build_trading_agent(cfg: Config, agent_type: str = "general") -> ReActAgent:
    llm = cfg.build_llm()
    tools = ToolRegistry()
    register_builtin_tools(tools)
    register_trading_tools(tools)

    # Optionally restrict tools by tag per agent type
    # data agent only gets data tools, execution agent gets execution tools, etc.
    tag_filter = {
        "data":      ["trading", "data", "onchain"],
        "analysis":  ["trading", "data", "analysis", "compute"],
        "execution": ["trading", "execution", "risk"],
        "general":   None,  # all tools
    }.get(agent_type)

    if tag_filter:
        # Override as_schema to only return tagged tools
        class FilteredRegistry(ToolRegistry):
            def as_schema(self, tags=None):
                return super().as_schema(tags=tag_filter)
        filtered = FilteredRegistry()
        filtered._tools = tools._tools
        tools = filtered

    memory = ShortTermMemory(max_messages=cfg.short_term_window)
    return ReActAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        system_prompt=TRADING_SYSTEM_PROMPT,
        max_iterations=cfg.max_iterations,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXAMPLES — run these to see the patterns in action
# ═══════════════════════════════════════════════════════════════════════════════

def example_single_agent():
    """Basic ReAct agent for a trading task."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single ReAct Agent")
    print("="*60)
    cfg = Config()
    agent = build_trading_agent(cfg)
    result = agent.run("What is the current BTC spot price and funding rate? Should I go long?")
    print(result.summary())


def example_with_reflection():
    """Agent + Reflection for high-stakes decisions."""
    print("\n" + "="*60)
    print("EXAMPLE 2: ReAct + Reflection")
    print("="*60)
    cfg = Config()
    agent = build_trading_agent(cfg)
    reflector = Reflector(cfg.build_llm(), max_retries=2)

    task = "Analyse BTC market structure and give a trade recommendation with entry, stop, and target."
    initial = agent.run(task)
    final = reflector.reflect_and_refine(agent, task, initial, verbose=True)
    print(f"\nFinal answer (score={final.score}/10):\n{final.final_answer}")


def example_orchestrator():
    """Orchestrator routing tasks to specialised sub-agents."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-Agent Orchestrator")
    print("="*60)
    cfg = Config()
    llm = cfg.build_llm()

    data_agent     = build_trading_agent(cfg, agent_type="data")
    analysis_agent = build_trading_agent(cfg, agent_type="analysis")
    execution_agent= build_trading_agent(cfg, agent_type="execution")

    orchestrator = Orchestrator(router_llm=llm)
    orchestrator.register("data",      data_agent,      "Fetches prices, OHLCV, on-chain data, funding rates")
    orchestrator.register("analysis",  analysis_agent,  "Analyses market structure, indicators, regime, sentiment")
    orchestrator.register("execution", execution_agent, "Sizes, risk-checks, and executes trade orders")

    result = orchestrator.run("Fetch the current ETH funding rate and spot price")
    print(f"Answer: {result.answer}")


def example_planner():
    """Planner + PlanExecutor for multi-step trading workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Planner → Execute Pipeline")
    print("="*60)
    cfg = Config()
    llm = cfg.build_llm()
    agent = build_trading_agent(cfg)

    planner = Planner(llm)
    plan = planner.create_plan(
        "Build and execute a complete BTC trade: data → analysis → decision → execution"
    )
    print(plan.display())

    executor = PlanExecutor(agent)
    completed_plan = executor.execute(
        plan,
        on_step_start=lambda s: print(f"\n--- Step {s.step}: {s.title} ---"),
        on_step_done=lambda s, r: print(f"    Done: {r.answer[:100]}"),
    )
    print("\n" + completed_plan.display())


if __name__ == "__main__":
    # Run all examples — comment out what you don't need
    example_single_agent()
    # example_with_reflection()
    # example_orchestrator()
    # example_planner()
