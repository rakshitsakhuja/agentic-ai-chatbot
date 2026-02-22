# Agentic AI Apps 🤖

> **Experimental Project**: This repository is a testing ground for exploring different AI agent capabilities and building practical use cases.

## Overview

This is an experimental project focused on:
- **Testing various agent capabilities** including planning, reasoning, reflection, and orchestration
- **Building real-world use cases** that leverage these agentic capabilities
- **Exploring different agent patterns** like ReAct, routing, and multi-agent systems

## Project Structure

### `my_agent/`
Core agent framework with modular components:
- **Agent Patterns**: Orchestrator, Planner, ReAct, Reflection, Router
- **LLM Integration**: Support for Anthropic and OpenAI models
- **Memory Systems**: Short-term and long-term memory capabilities
- **Tools**: Extensible tool system (ArXiv, RAG, NL-to-SQL, and more)
- **Observability**: Built-in tracing for debugging and monitoring

### `chatbot_app/`
Interactive chatbot application showcasing agent capabilities with a user-friendly interface.

## Use Cases & Examples

Explore practical implementations in the `my_agent/examples/` directory:
- **NL-to-SQL**: Natural language database querying
- **RAG**: Retrieval-Augmented Generation for document Q&A
- **Trading Agent**: Autonomous trading decision-making

## Getting Started

```bash
# Install dependencies for the agent framework
cd my_agent
pip install -r requirements.txt

# Install dependencies for the chatbot app
cd chatbot_app
pip install -r requirements.txt
```

## Experimentation Goals

- Test different prompting strategies and agent architectures
- Evaluate performance across various LLM providers
- Build robust tooling for agent interactions
- Create reproducible patterns for common agentic workflows
- Explore memory and context management techniques

## Status

🚧 **Active Development** - This project is under active experimentation. APIs and architectures may change as we explore what works best.

