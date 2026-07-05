#!/usr/bin/env python
"""
verify_rag_pipeline.py

A unified verification script for VectorDB.
Tests the search algorithms, distance metrics, chunking logic, and the FastAPI
RAG endpoints (using mocked Ollama API responses). Also probes for a live Ollama
instance to test real online document embedding and RAG if available.
"""

import sys
import asyncio
import json
import math
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import project modules
try:
    from database import (
        cosine_distance,
        euclidean_distance,
        manhattan_distance,
        chunk_text,
        VectorDB,
        DocumentDB
    )
    from algorithms import HNSW, KDTree, BruteForce
    from app import app, doc_insert, doc_ask, ollama, ollama_status, doc_db
except ImportError as e:
    print(f"\033[91mError: Could not import project files. Make sure to run this script in the root directory of the project. Details: {e}\033[0m")
    sys.exit(1)


# Colors for formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title):
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")


def run_unit_tests():
    print_header("1. RUNNING CORE UNIT TESTS")

    tests_failed = 0

    # Test Distance Metrics
    print("Testing distance metrics...", end=" ")
    try:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        c = [1.0, 0.0, 0.0]

        # Cosine distance
        assert abs(cosine_distance(a, b) - 1.0) < 1e-7
        assert abs(cosine_distance(a, c) - 0.0) < 1e-7

        # Euclidean distance
        assert abs(euclidean_distance(a, b) - math.sqrt(2.0)) < 1e-7
        assert abs(euclidean_distance(a, c) - 0.0) < 1e-7

        # Manhattan distance
        assert abs(manhattan_distance(a, b) - 2.0) < 1e-7
        assert abs(manhattan_distance(a, c) - 0.0) < 1e-7

        print(f"{GREEN}PASSED{RESET}")
    except Exception as e:
        print(f"{RED}FAILED ({e}){RESET}")
        tests_failed += 1

    # Test Text Chunking
    print("Testing text chunking helper...", end=" ")
    try:
        sample_text = "one two three four five six seven eight nine ten"
        # chunk_size=3, overlap=1
        # Chunks should be:
        # 1: "one two three"
        # 2: "three four five"
        # 3: "five six seven"
        # 4: "seven eight nine"
        # 5: "nine ten"
        chunks = chunk_text(sample_text, chunk_size=3, overlap=1)
        assert len(chunks) == 5, f"Expected 5 chunks, got {len(chunks)}"
        assert chunks[0] == "one two three"
        assert chunks[1] == "three four five"
        assert chunks[-1] == "nine ten"
        print(f"{GREEN}PASSED{RESET}")
    except Exception as e:
        print(f"{RED}FAILED ({e}){RESET}")
        tests_failed += 1

    # Test Search Algorithms Correctness
    print("Testing Search Algorithms (Brute Force, KD-Tree, HNSW)...", end=" ")
    try:
        # Let's verify KDTree and HNSW retrieve the correct nearest neighbor
        # 3D points
        points = [
            ("p1", [0.1, 0.2, 0.3]),
            ("p2", [0.9, 0.8, 0.7]),
            ("p3", [0.5, 0.5, 0.5]),
        ]
        query = [0.12, 0.18, 0.31] # Very close to p1

        # Brute Force
        bf = BruteForce()
        bf_res = bf.search(query, points, k=1, distance_fn=euclidean_distance)
        assert bf_res[0][0] == "p1", f"BruteForce expected p1, got {bf_res[0][0]}"

        # KDTree
        kdt = KDTree(dim=3)
        kdt.build(points)
        kdt_res = kdt.search(query, k=1, distance_fn=euclidean_distance)
        assert kdt_res[0][0] == "p1", f"KDTree expected p1, got {kdt_res[0][0]}"

        # HNSW
        hnsw = HNSW(dim=3, M=4, ef_construction=10)
        for pid, vec in points:
            hnsw.insert(pid, vec)
        hnsw_res = hnsw.search(query, k=1, distance_fn=euclidean_distance, ef=10)
        assert hnsw_res[0][0] == "p1", f"HNSW expected p1, got {hnsw_res[0][0]}"

        print(f"{GREEN}PASSED{RESET}")
    except Exception as e:
        print(f"{RED}FAILED ({e}){RESET}")
        tests_failed += 1

    if tests_failed == 0:
        print(f"{GREEN}{BOLD}All Core Unit Tests Passed!{RESET}")
        return True
    else:
        print(f"{RED}{BOLD}{tests_failed} Unit Tests Failed!{RESET}")
        return False


async def run_mocked_api_tests():
    print_header("2. RUNNING MOCKED API ENDPOINT TESTS")
    tests_failed = 0

    # Test /doc/insert Endpoint (Mocked)
    print("Testing /doc/insert endpoint logic...", end=" ")
    try:
        # Create a mock Request
        mock_req = MagicMock()
        mock_req.json = AsyncMock(return_value={
            "title": "Verifying Vector DB Features",
            "text": "This text will be split into overlapping chunks and then each chunk will be embedded using nomic-embed-text. The resulting vectors are inserted into the local HNSW database for instant semantic search. RAG will retrieve these chunks later."
        })

        # Clear document DB first
        global doc_db
        doc_db.docs.clear()
        doc_db.hnsw = HNSW(dim=768, M=16, ef_construction=200)
        doc_db._next_id = 0

        # Patch Ollama calls
        with patch.object(ollama, 'is_online', return_value=True), \
             patch.object(ollama, 'embed', return_value=[0.05] * 768) as mock_embed:

            res = await doc_insert(mock_req)
            
            # Checks
            assert res['status'] == 'ok', "Insert failed status check"
            assert res['title'] == "Verifying Vector DB Features"
            assert res['chunks'] > 0, "No chunks were generated"
            assert doc_db.get_count() == res['chunks'], "Database count mismatch"
            assert mock_embed.call_count == res['chunks'], "Incorrect number of embedding calls"
            
            first_doc_id = res['ids'][0]
            assert doc_db.docs[first_doc_id]['title'] == "Verifying Vector DB Features"
            assert len(doc_db.docs[first_doc_id]['embedding']) == 768

            print(f"{GREEN}PASSED (Created {res['chunks']} chunks with 768D embeddings){RESET}")
    except Exception as e:
        print(f"{RED}FAILED ({e}){RESET}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

    # Test /doc/ask Endpoint (Mocked RAG Pipeline)
    print("Testing /doc/ask RAG pipeline logic...", end=" ")
    try:
        # Create mock Request
        mock_req = MagicMock()
        mock_req.json = AsyncMock(return_value={
            "question": "What is being verified?",
            "k": 1
        })

        # Patch Ollama online check, embedding question, and streaming response
        mock_tokens = ["Based", " on", " the", " context,", " Vector", " DB", " features", " are", " being", " verified."]
        
        def mock_generator(prompt, model='llama3.2'):
            for token in mock_tokens:
                yield token

        with patch.object(ollama, 'is_online', return_value=True), \
             patch.object(ollama, 'embed', return_value=[0.05] * 768), \
             patch.object(ollama, 'generate_stream', side_effect=mock_generator) as mock_gen:

            # Call endpoint
            response = await doc_ask(mock_req)
            
            # Read streaming response chunks
            chunks_received = []
            async for chunk in response.body_iterator:
                # SSE lines look like: data: {"token": "..."}\n\n
                if isinstance(chunk, bytes):
                    line = chunk.decode('utf-8').strip()
                else:
                    line = chunk.strip()
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    data = json.loads(data_str)
                    chunks_received.append(data)

            # Analyze streamed events
            tokens = [c['token'] for c in chunks_received if 'token' in c]
            final_metadata = [c for c in chunks_received if 'done' in c][0]

            assert len(tokens) == len(mock_tokens), "Mismatched token length"
            assert "".join(tokens) == "".join(mock_tokens), "Tokens do not match expected stream"
            assert final_metadata['done'] is True, "Missing done flag"
            assert len(final_metadata['context']) == 1, "Context chunk count mismatch"
            assert final_metadata['context'][0]['title'] == "Verifying Vector DB Features", "Context title mismatch"

            print(f"{GREEN}PASSED (RAG successfully retrieved context and streamed answer){RESET}")
    except Exception as e:
        print(f"{RED}FAILED ({e}){RESET}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

    if tests_failed == 0:
        print(f"{GREEN}{BOLD}All Mocked API Tests Passed!{RESET}")
        return True
    else:
        print(f"{RED}{BOLD}{tests_failed} API Tests Failed!{RESET}")
        return False


async def run_live_tests(active_llm, embed_model):
    # 1. Test live document embedding
    print("Testing live document embedding...", end=" ")
    try:
        # Clear document DB
        global doc_db
        doc_db.docs.clear()
        doc_db.hnsw = HNSW(dim=768, M=16, ef_construction=200)
        doc_db._next_id = 0

        # Construct a request manually
        mock_req = MagicMock()
        mock_req.json = AsyncMock(return_value={
            "title": "Ollama Live Test Doc",
            "text": "The solar system consists of the Sun and the objects that orbit it. There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006."
        })

        # Call real doc_insert (no patching)
        res = await doc_insert(mock_req)
        assert res['status'] == 'ok'
        assert res['chunks'] > 0
        print(f"{GREEN}PASSED! Created {res['chunks']} chunks with real embeddings.{RESET}")
    except Exception as e:
        print(f"{RED}FAILED! Check if '{embed_model}' is fully downloaded. Error: {e}{RESET}")
        return

    # 2. Test live RAG generation
    print("Testing live RAG pipeline...", end=" ")
    try:
        mock_req = MagicMock()
        # Ask something related to the doc we just inserted
        mock_req.json = AsyncMock(return_value={
            "question": "How many planets are in our solar system?",
            "k": 1
        })

        original_generate_stream = ollama.generate_stream
        def custom_generate_stream(prompt, model='llama3.2'):
            return original_generate_stream(prompt, model=active_llm)
        
        with patch.object(ollama, 'generate_stream', side_effect=custom_generate_stream):
            response = await doc_ask(mock_req)

        # Stream response
        print(f"\n{BOLD}AI Answer:{RESET} ", end="", flush=True)
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                line = chunk.decode('utf-8').strip()
            else:
                line = chunk.strip()
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                data = json.loads(data_str)
                if 'token' in data:
                    print(data['token'], end="", flush=True)
        print()
        print(f"{GREEN}PASSED! Live RAG generated response successfully.{RESET}")
    except Exception as e:
        print(f"{RED}FAILED! Error: {e}{RESET}")


def probe_and_run_online_tests():
    print_header("3. PROBING FOR LIVE OLLAMA INSTANCE")
    
    online = ollama.is_online()
    if not online:
        print(f"{YELLOW}Ollama is currently OFFLINE at {ollama.base_url}.{RESET}")
        print(f"To perform live integration testing:")
        print(f"  1. Start Ollama: {BOLD}ollama serve{RESET}")
        print(f"  2. Pull required models:")
        print(f"     {BOLD}ollama pull nomic-embed-text{RESET}")
        print(f"     {BOLD}ollama pull llama3.2:1b{RESET}  (or llama3.2)")
        print(f"  3. Rerun this script.")
        return

    print(f"{GREEN}Ollama is ONLINE at {ollama.base_url}!{RESET}")

    # Check available models
    status = asyncio.run(ollama_status())
    models = status.get('models', [])
    print(f"Installed Ollama models: {models}")

    embed_model = 'nomic-embed-text'
    llm_model = 'llama3.2'
    
    # Check if nomic-embed-text is available
    has_embed = any(embed_model in m for m in models)
    # Check if llama3.2 (or llama3.2:1b, etc.) is available
    has_llm = any('llama3.2' in m for m in models)

    if not has_embed:
        print(f"{RED}Error: Model '{embed_model}' is not pulled in Ollama.{RESET}")
        print(f"Please run: {BOLD}ollama pull {embed_model}{RESET}")
        return

    # Use llama3.2:1b or standard llama3.2 depending on what is installed
    active_llm = None
    for m in models:
        if 'llama3.2' in m:
            active_llm = m
            break

    if not active_llm:
        print(f"{YELLOW}Warning: Model '{llm_model}' is not pulled in Ollama. RAG model text generation will fail.{RESET}")
        print(f"Please run: {BOLD}ollama pull {llm_model}{RESET} (or llama3.2:1b)")
        return
    else:
        print(f"Using LLM model for RAG: {active_llm}")

    print_header("4. RUNNING LIVE ONLINE TESTS")
    asyncio.run(run_live_tests(active_llm, embed_model))


if __name__ == "__main__":
    print(f"{BOLD}==================================================={RESET}")
    print(f"{BOLD}    VECTOR DB RAG PIPELINE VERIFICATION TOOL       {RESET}")
    print(f"{BOLD}==================================================={RESET}")

    unit_ok = run_unit_tests()
    if not unit_ok:
        sys.exit(1)

    mock_ok = asyncio.run(run_mocked_api_tests())
    if not mock_ok:
        sys.exit(1)

    probe_and_run_online_tests()
    print(f"\n{BOLD}{GREEN}Verification tool execution completed successfully!{RESET}\n")
