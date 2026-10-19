# VectorDB from Scratch

A small vector database written in Python, with a web UI. It implements **HNSW**, **KD-Tree** and **Brute Force** search side by side, plus a **RAG pipeline** that uses a local LLM through Ollama.

I built it to learn how vector databases index and search embeddings. The search algorithms in `algorithms.py` use only the Python standard library. PCA in `pca.py` is written by hand on top of NumPy.

---

## What It Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (approximate), KD-Tree and Brute Force (exact). Run all three on the same query and compare their speed |
| **3 Distance Metrics** | Cosine, Euclidean and Manhattan distance |
| **16D Demo Vectors** | 20 hand-made vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | The demo vectors projected to 2D, so you can see the category clusters |
| **Document Embedding** | Paste any text and Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask a question about your documents. HNSW retrieves the context and a local LLM answers |
| **REST API** | Endpoints for insert, delete, search, benchmark and HNSW graph info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

---

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com) with two models: `nomic-embed-text` (embeddings) and `llama3.2` (answers)

The Search tab and its endpoints work without Ollama. The Documents and Ask AI tabs need it.

## Run

```bash
git clone https://github.com/Uditp11/vectordb-from-scratch.git
cd vectordb-from-scratch
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull llama3.2

python app.py
```

Start Ollama with `ollama serve` if it is not already running. Then open http://localhost:8080.

## Verification Script

```bash
python verify_rag_pipeline.py
```

`verify_rag_pipeline.py` is a verification script, not a test suite. It checks the distance metrics, the chunking helper and a small search example, then calls the document insert and RAG endpoints with Ollama mocked out. If it finds a live Ollama with the two models pulled, it also runs the real embedding and RAG flow.

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Paste a 16-dimensional vector into the input (for example, copy one from the scatter plot)
- Choose the algorithm: **HNSW**, **KD-Tree** or **Brute Force**
- Choose the distance metric: **Cosine**, **Euclidean** or **Manhattan**
- Click **SEARCH** to see the results in a table, with the matching points highlighted on the scatter plot
- Click **COMPARE ALL** to run all three algorithms and compare their speed

The scatter plot shows all 20 vectors projected to 2D with PCA. The 4 categories form separate clusters.

### Tab 2: Documents (Real Embeddings)

1. Type a title (for example `Operating Systems Notes`)
2. Paste any text, such as lecture notes or an article
3. Click **EMBED & INSERT**
4. Long documents are split into overlapping chunks of 250 words
5. Each chunk gets its own embedding and is stored in the HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Insert some documents in Tab 2 first
2. Type a question about them
3. Click **ASK**

What happens:
```
1. Your question is embedded with nomic-embed-text (768D vector)
2. HNSW search finds the 3 most similar chunks
3. Those chunks are sent as context to llama3.2
4. llama3.2 answers based only on that context
```

The answer streams in token by token. Click the context chips to see which chunks were used.

---

## REST API Reference

The server listens on `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | k nearest neighbors search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/{id}` | Delete by ID |
| `GET` | `/items` | List all demo vectors and their PCA coordinates |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Run one query on all 3 algorithms with timings |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document and RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store a document |
| `GET` | `/doc/list` | none | List all stored chunks |
| `DELETE` | `/doc/delete/{id}` | none | Delete one chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve and generate (SSE stream) |
| `GET` | `/status` | none | Ollama status and model info |

---

## Project Structure

```
vectordb-from-scratch/
├── algorithms.py          ← Search algorithms: BruteForce, KD-Tree, HNSW
├── pca.py                 ← PCA projection to 2D (covariance and eigendecomposition with NumPy)
├── database.py            ← VectorDB (16D demo) + DocumentDB (768D) + distance metrics
├── app.py                 ← FastAPI server + Ollama client + REST endpoints
├── index.html             ← Frontend UI (Chart.js scatter plot, RAG chat)
├── verify_rag_pipeline.py ← Verification script (see above)
├── requirements.txt       ← Python dependencies
└── README.md              ← This file
```

### Architecture

```
BruteForce          O(N·d) per query    Exact, baseline
KDTree              O(log N) at best    Exact for Euclidean and Manhattan, axis-aligned partitioning
HNSW                about O(log N)      Approximate, multilayer small-world graph

VectorDB            One interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client for /api/embeddings and /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node is randomly assigned a maximum layer. Layer 0 has all nodes with many connections. Higher layers have exponentially fewer nodes with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from the node's assigned max down to 0, run a beam search (ef_construction=200) and connect to the M nearest neighbors in both directions.

**Search:** The same greedy descent from the top layer. At layer 0, expand to the ef nearest candidates using a priority queue.

**Why it is fast:** The upper layers act like a highway. You quickly reach the right neighborhood, then search it closely at layer 0.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension, cycling through all dimensions. Search skips a whole subtree when the closest possible point in it cannot beat the current best.

That skip compares the distance to the splitting plane with the current best distance. This is a valid bound for Euclidean and Manhattan distance, so KD-Tree search is exact for those two. It is not a valid bound for cosine distance, so with cosine the KD-Tree can return different results from Brute Force.

**Weakness:** It slows down in high dimensions (the curse of dimensionality). It works best in low dimensions.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions almost all of the space is near a boundary, so almost no subtrees get pruned. HNSW's graph approach does not have this problem.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in the status bar | Run `ollama serve` in a terminal |
| The first embedding is slow | Ollama loads the model into memory on first use |
| Port 8080 already in use | Find the process with `netstat -ano \| findstr 8080`, then `taskkill /PID <pid> /F` (Windows) |
| The LLM answer is slow | Expected on a laptop CPU. Try the smaller model below |

### Use a Smaller LLM

```bash
ollama pull llama3.2:1b
```

Then in `app.py` change the default `model='llama3.2'` in `OllamaClient.generate_stream()` to `model='llama3.2:1b'` and restart the server.

---

## License

MIT
