# ScholarAI Writer Agent

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Orchestrated-blue" alt="LangGraph Orchestrated">
  <img src="https://img.shields.io/badge/LLM-Groq-green" alt="LLM Groq">
  <img src="https://img.shields.io/badge/Research-Automated-orange" alt="Research Automated">
</p>

<p align="center">
  <b>An autonomous research-driven technical blog generation agent built using structured planning, routing, and multi-step LLM orchestration.</b>
</p>

## 🚀 Overview

ScholarAI Writer Agent is a LangGraph-powered autonomous writing system that transforms a single topic query into a fully structured, well-researched, multi-section technical blog post.

**It is not a simple prompt wrapper.** It performs:

- **Routing** → Determines whether research is required
- **Web Research** (if needed) → Retrieves and deduplicates evidence
- **Structured Planning** → Generates a high-quality blog blueprint
- **Parallel Section Writing** → Writes sections independently
- **Reduction & Compilation** → Produces a final polished Markdown article

This architecture ensures grounded, structured, and production-quality long-form outputs.

```mermaid
graph TD
    Start([START]) --> Router
    
    Router --> ResearchMode{Research Mode}
    ResearchMode -->|Closed Book / Hybrid / Open Book| Research[Research Node<br/><i>Web search & evidence extraction</i>]
    
    Research --> Orchestrator[Orchestrator<br/><i>Creates structured plan</i>]
    
    Orchestrator --> FanOut[Fan-out]
    FanOut --> Worker1[Worker Node 1<br/><i>Section 1</i>]
    FanOut --> Worker2[Worker Node 2<br/><i>Section 2</i>]
    FanOut --> Worker3[Worker Node 3<br/><i>Section 3</i>]
    FanOut --> WorkerN[Worker Node N<br/><i>Section N</i>]
    
    Worker1 --> Reducer
    Worker2 --> Reducer
    Worker3 --> Reducer
    WorkerN --> Reducer
    
    Reducer[Reducer<br/><i>Merge + Export Markdown</i>] --> End([END])
    
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef node fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px;
    
    class Router,Research,Orchestrator,Reducer process;
    class ResearchMode decision;
    class Worker1,Worker2,Worker3,WorkerN node;
```

### Graph-Oriented Orchestration

Built using:
- **LangGraph** for deterministic state transitions
- **LangChain** structured output parsing
- **Groq** LLM inference
- **Tavily API** for external research grounding

## 🏗 Core Components

### 1️⃣ Router Node
Determines:
- Does this topic require fresh web research?
- Is it evergreen (closed-book) or time-sensitive (open-book)?

### 2️⃣ Research Node
- Executes high-signal web queries
- Deduplicates URLs
- Extracts structured evidence objects
- Avoids hallucinated dates or claims

### 3️⃣ Orchestrator
Produces a strongly-typed Plan object:
- 5–9 structured sections
- Explicit goals
- Concrete bullet breakdowns
- Target word counts
- Flags for citations / code inclusion

### 4️⃣ Worker Nodes (Parallelized)
Each worker:
- Writes exactly one section
- Follows bullet structure strictly
- Respects research/citation rules
- Includes code when required

### 5️⃣ Reducer
- Orders sections
- Compiles final Markdown
- Writes output file automatically

## 🔍 Intelligent Research Modes

| Mode | Description |
|------|-------------|
| `closed_book` | Evergreen conceptual topics |
| `hybrid` | Mix of evergreen + recent tools/models |
| `open_book` | News, rankings, recent releases |

The agent automatically decides the mode before planning.

## 📦 Example Usage

```python
out = run("Write an Essay on the newly published research paper about VL-JEPA by Yann LeCun.")
print(out["final"])
```

**Output:**
- Structured Markdown blog
- Grounded citations (if needed)
- Multi-section format
- Saved automatically as `.md`



