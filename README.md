# KG-RAG Live Demo

Four small demos of **Knowledge Graph RAG** (Retrieval-Augmented Generation): instead of retrieving text chunks from a vector store, answers are grounded in facts retrieved from a graph.

| Demo | Script | Graph store | How the graph is built |
|------|--------|-------------|------------------------|
| 1. FAQ Graph RAG | `src/KG_RAG_Demo.py` | Neo4j Aura (cloud) | Parses `Q:`/`A:` pairs from `data/FAQ.txt` |
| 2. Entity Graph RAG | `src/KG_Demo.py` | NetworkX (in-memory) | LLM extracts entities and relationships from `data/sample.txt` |
| 3. RAG Evaluation | `src/evals_demo.py` | – | Scores a RAG answer with DeepEval (Faithfulness, Answer Relevancy) |
| 4. Guardrails | `src/guardrails_demo.py` | – | Blocks off-topic questions with NeMo Guardrails wrapped around a LangChain chain |

## Project structure

```
KG_RAG_Live_Demo/
├── src/
│   ├── KG_RAG_Demo.py   # Demo 1: FAQ -> Neo4j -> interactive chat
│   ├── KG_Demo.py       # Demo 2: text -> LLM graph extraction -> NetworkX -> RAG answer
│   ├── evals_demo.py    # Demo 3: DeepEval metrics on a RAG answer
│   └── guardrails_demo.py  # Demo 4: NeMo Guardrails + LangChain
├── data/
│   ├── FAQ.txt          # Q/A pairs for Demo 1
│   └── sample.txt       # Prose about a fictional company for Demo 2
├── .env.example         # Template for required environment variables
├── requirements.txt
└── README.md
```

## How it works

### Demo 1 – `KG_RAG_Demo.py` (Neo4j)

```
FAQ.txt -> parse -> Neo4j graph -> keyword search (Cypher) -> OpenAI LLM -> answer
```

Graph model: `(Topic)-[:HAS_FAQ]->(FAQ)-[:HAS_ANSWER]->(Answer)`

1. Parses `data/FAQ.txt` into question/answer pairs.
2. Connects to Neo4j Aura, **clears the existing graph**, and loads the FAQs.
3. For each user question, extracts keywords and finds matching FAQ/Topic nodes with Cypher `CONTAINS`.
4. Passes the retrieved entries to the LLM (via a LangChain prompt) and prints an answer restricted to that context.

### Demo 2 – `KG_Demo.py` (NetworkX)

```
sample.txt -> split into chunks -> LLMGraphTransformer -> NetworkX DiGraph -> traverse -> LLM -> answer
```

1. Loads `data/sample.txt` and splits it into 500-token chunks.
2. `LLMGraphTransformer` extracts entities (nodes) and relationships (edges).
3. Builds an in-memory `networkx.DiGraph` and prints its nodes.
4. Collects the outgoing relationships of `Technova Corp` as context and asks the LLM a question about them.

### Demo 3 – `evals_demo.py` (DeepEval)

Runs a small LangChain RAG chain on a hard-coded question and context ("Who founded TechNova Corp?"), then scores the answer with `FaithfulnessMetric` and `AnswerRelevancyMetric` (pass threshold 0.7). The judge model is also an OpenAI model, so it makes extra API calls. DeepEval writes a local `.deepeval/` cache (git-ignored).

### Demo 4 – `guardrails_demo.py` (NeMo Guardrails)

Defines Colang rules inline that refuse off-topic questions (cooking, politics, sports), wraps a LangChain chain with `RunnableRails`, and runs one allowed and one blocked query.

## Setup

Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```

Fill in `.env`:

| Variable | Needed by | Description |
|----------|-----------|-------------|
| `OPENAI_API_KEY` | all demos | OpenAI API key |
| `NEO4J_URI` | Demo 1 | e.g. `neo4j+s://xxxx.databases.neo4j.io` (from [console.neo4j.io](https://console.neo4j.io)) |
| `NEO4J_USERNAME` | Demo 1 | Usually `neo4j` |
| `NEO4J_PASSWORD` | Demo 1 | Aura instance password |
| `NEO4J_DATABASE` | Demo 1 | Optional, defaults to `neo4j` |

## Run

```bash
python src/KG_RAG_Demo.py   # interactive FAQ chat (type quit/exit to stop)
python src/KG_Demo.py       # one-shot entity-graph demo
python src/evals_demo.py    # DeepEval scoring
python src/guardrails_demo.py  # guardrails allowed/blocked test
```

Example questions for Demo 1: "What is Neo4j?", "Tell me about RAG", "Explain knowledge graph".

## Notes

- Demo 1 runs `MATCH (n) DETACH DELETE n` on startup. **Use an empty/dedicated Neo4j instance.**
- Both scripts use the model name `gpt-5.4-mini`; change it in the scripts if your account uses a different model.
- Retrieval in Demo 1 is simple keyword matching, not semantic search. The demos favour clarity over production robustness.
- To use your own data, edit `data/FAQ.txt` (keep the `Q: ` / `A: ` format) or `data/sample.txt`.
