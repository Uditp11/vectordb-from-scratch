"""
app.py — FastAPI backend for the VectorDB project.

Serves the frontend, exposes REST endpoints for:
  • Demo vector search / insert / delete / benchmark
  • Document RAG (chunk → embed → store → ask with streaming LLM)
  • Ollama integration for embeddings and text generation

Run:  python app.py
"""

import json
import os
import time

import requests
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from database import VectorDB, DocumentDB, chunk_text, get_distance_fn


# =====================================================================
# ASCII Banner
# =====================================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗        ║
║   ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗       ║
║   ██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝       ║
║   ╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗       ║
║    ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║       ║
║     ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝       ║
║                    ██████╗ ██████╗                            ║
║                    ██╔══██╗██╔══██╗                           ║
║                    ██║  ██║██████╔╝                           ║
║                    ██║  ██║██╔══██╗                           ║
║                    ██████╔╝██████╔╝                           ║
║                    ╚═════╝ ╚═════╝                            ║
║                                                              ║
║   Python Edition — From-Scratch Vector Database              ║
║   Algorithms: BruteForce · KD-Tree · HNSW                   ║
║   Server: http://localhost:8080                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# =====================================================================
# Ollama Client
# =====================================================================

class OllamaClient:
    """
    Lightweight HTTP client for the Ollama local LLM server.

    Provides embedding, generation, and streaming generation.
    """

    def __init__(self, base_url='http://localhost:11434'):
        self.base_url = base_url.rstrip('/')

    def is_online(self):
        """Check whether the Ollama server is reachable."""
        try:
            r = requests.get(self.base_url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text, model='nomic-embed-text'):
        """
        Get an embedding vector for the given text.

        Uses the Ollama /api/embeddings endpoint.

        Returns
        -------
        list[float]
            The embedding vector.

        Raises
        ------
        RuntimeError
            If the request fails.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {'model': model, 'prompt': text}
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get('embedding', [])
        except Exception as e:
            raise RuntimeError(f"Ollama embed failed: {e}")

    def generate(self, prompt, model='llama3.2', stream=False):
        """
        Generate a non-streaming response from the LLM.

        Returns
        -------
        str
            The generated text.
        """
        url = f"{self.base_url}/api/generate"
        payload = {'model': model, 'prompt': prompt, 'stream': stream}
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get('response', '')
        except Exception as e:
            raise RuntimeError(f"Ollama generate failed: {e}")

    def generate_stream(self, prompt, model='llama3.2'):
        """
        Stream tokens from the LLM as a generator.

        Yields
        ------
        str
            One token at a time.
        """
        url = f"{self.base_url}/api/generate"
        payload = {'model': model, 'prompt': prompt, 'stream': True}
        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get('response', '')
                        if token:
                            yield token
                        if data.get('done', False):
                            break
        except Exception as e:
            yield f"\n[Error: {e}]"


# =====================================================================
# FastAPI Application
# =====================================================================

app = FastAPI(title="VectorDB", version="1.0.0")

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate databases and Ollama client
vector_db = VectorDB()
doc_db = DocumentDB()
ollama = OllamaClient()

# Resolve the directory this script lives in (for serving index.html)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Static / Frontend ───────────────────────────────────────────────

@app.get("/")
async def serve_root():
    """Serve index.html at the root URL."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/index.html")
async def serve_index():
    """Serve index.html explicitly."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


# ── Demo Vector Endpoints ───────────────────────────────────────────

@app.get("/search")
async def search(
    v: str = Query(..., description="Comma-separated float vector"),
    k: int = Query(5),
    metric: str = Query('cosine'),
    algo: str = Query('hnsw'),
):
    """
    Search for k nearest neighbors.

    Query params:
      v     — comma-separated vector values (e.g. "0.5,0.3,0.1,...")
      k     — number of results
      metric — cosine | euclidean | manhattan
      algo  — brute | kdtree | hnsw
    """
    try:
        query = [float(x) for x in v.split(',')]
    except ValueError:
        return JSONResponse({'error': 'Invalid vector format'}, status_code=400)

    try:
        results = vector_db.search(query, k=k, metric=metric, algo=algo)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

    return {'results': results}


@app.post("/insert")
async def insert_item(request: Request):
    """
    Insert a new vector.

    JSON body: {id, label, category, vector}
    """
    body = await request.json()
    required = ['id', 'label', 'category', 'vector']
    for key in required:
        if key not in body:
            return JSONResponse({'error': f'Missing field: {key}'}, status_code=400)

    vector_db.insert(body['id'], body['label'], body['category'], body['vector'])
    return {'status': 'ok', 'id': body['id']}


@app.delete("/delete/{item_id}")
async def delete_item(item_id: str):
    """Delete a vector by id."""
    deleted = vector_db.delete(item_id)
    if not deleted:
        return JSONResponse({'error': 'Item not found'}, status_code=404)
    return {'status': 'ok', 'id': item_id}


@app.get("/items")
async def get_items():
    """Return all items plus their PCA 2D coordinates."""
    items = vector_db.get_items()
    pca_coords = vector_db.get_pca_coords()
    return {'items': items, 'pca_coords': pca_coords}


@app.get("/benchmark")
async def benchmark(
    v: str = Query(..., description="Comma-separated float vector"),
    k: int = Query(5),
    metric: str = Query('cosine'),
):
    """
    Benchmark all three algorithms on the same query.

    Returns timing (ms) and results for each algorithm.
    """
    try:
        query = [float(x) for x in v.split(',')]
    except ValueError:
        return JSONResponse({'error': 'Invalid vector format'}, status_code=400)

    try:
        results = vector_db.benchmark(query, k=k, metric=metric)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

    return results


@app.get("/hnsw-info")
async def hnsw_info():
    """Return HNSW graph diagnostics."""
    return vector_db.get_hnsw_info()


@app.get("/stats")
async def stats():
    """Return database summary statistics."""
    return vector_db.get_stats()


# ── Document / RAG Endpoints ────────────────────────────────────────

@app.post("/doc/insert")
async def doc_insert(request: Request):
    """
    Insert a document: chunk the text, embed each chunk via Ollama,
    and store in the document database.

    JSON body: {title, text}
    """
    body = await request.json()
    title = body.get('title', 'Untitled')
    text = body.get('text', '')

    if not text.strip():
        return JSONResponse({'error': 'Empty document text'}, status_code=400)

    # Check Ollama availability
    if not ollama.is_online():
        return JSONResponse(
            {'error': 'Ollama is not running. Start it with: ollama serve'},
            status_code=503,
        )

    # Chunk the text
    chunks = chunk_text(text, chunk_size=250, overlap=50)
    if not chunks:
        return JSONResponse({'error': 'Could not chunk text'}, status_code=400)

    # Embed each chunk
    embeddings = []
    for chunk in chunks:
        try:
            emb = ollama.embed(chunk)
            embeddings.append(emb)
        except RuntimeError as e:
            return JSONResponse({'error': str(e)}, status_code=500)

    # Store
    inserted_ids = doc_db.insert_chunks(title, chunks, embeddings)
    return {
        'status': 'ok',
        'title': title,
        'chunks': len(chunks),
        'ids': inserted_ids,
    }


@app.get("/doc/list")
async def doc_list():
    """Return all stored document chunks."""
    return {'docs': doc_db.get_docs(), 'count': doc_db.get_count()}


@app.delete("/doc/delete/{doc_id}")
async def doc_delete(doc_id: str):
    """Delete a document chunk by id."""
    deleted = doc_db.delete(doc_id)
    if not deleted:
        return JSONResponse({'error': 'Document not found'}, status_code=404)
    return {'status': 'ok', 'id': doc_id}


@app.post("/doc/ask")
async def doc_ask(request: Request):
    """
    RAG endpoint: embed the question, retrieve relevant chunks,
    build a prompt, and stream the LLM response.

    JSON body: {question, k (optional, default 3)}

    Returns Server-Sent Events (SSE) with tokens as they are generated.
    """
    body = await request.json()
    question = body.get('question', '')
    k = body.get('k', 3)

    if not question.strip():
        return JSONResponse({'error': 'Empty question'}, status_code=400)

    if not ollama.is_online():
        return JSONResponse(
            {'error': 'Ollama is not running. Start it with: ollama serve'},
            status_code=503,
        )

    # Embed the question
    try:
        query_emb = ollama.embed(question)
    except RuntimeError as e:
        return JSONResponse({'error': str(e)}, status_code=500)

    # Retrieve relevant chunks
    results = doc_db.search(query_emb, k=k)

    if not results:
        async def no_context():
            msg = json.dumps({'token': 'No documents found. Please insert documents first.'})
            yield f"data: {msg}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(no_context(), media_type='text/event-stream')

    # Build the RAG prompt
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{r['title']} — chunk {r['chunk_index']}]\n{r['text']}")
    context = '\n\n'.join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the "
        "following context. If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # Build context metadata for the frontend
    context_meta = [
        {'title': r['title'], 'text': r['text'], 'chunk_index': r['chunk_index']}
        for r in results
    ]

    # Stream the LLM response as SSE
    async def event_stream():
        for token in ollama.generate_stream(prompt):
            data = json.dumps({'token': token})
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'done': True, 'context': context_meta})}\n\n"

    return StreamingResponse(event_stream(), media_type='text/event-stream')


# ── Ollama Status ───────────────────────────────────────────────────

@app.get("/status")
async def ollama_status():
    """Check Ollama connectivity and return model info."""
    online = ollama.is_online()
    models = []
    if online:
        try:
            r = requests.get(f"{ollama.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                models = [m['name'] for m in data.get('models', [])]
        except Exception:
            pass

    return {
        'ollama_online': online,
        'models': models,
        'embed_model': 'nomic-embed-text',
        'llm_model': 'llama3.2',
    }


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == '__main__':
    import sys
    # Ensure UTF-8 output on Windows terminals
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    try:
        print(BANNER)
    except UnicodeEncodeError:
        print("\n=== VectorDB — Python Edition ===\n")
    uvicorn.run(app, host='0.0.0.0', port=8080)
