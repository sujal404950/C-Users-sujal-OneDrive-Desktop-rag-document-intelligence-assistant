"""
RAG Document Intelligence Assistant - Backend API tests.

Covers:
 - Health check
 - Session creation
 - Document upload (TXT + CSV)
 - Session state retrieval (chunks/documents)
 - Ask (real Gemini call, verifies planted fact appears in answer)
 - Evaluate (real Gemini calls for benchmark questions)
 - Error handling (ask without documents, unsupported extension)
"""
import os
import io
import pytest
import requests

BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")

PLANTED_FACT_DOC = """Project Bluefin Research Notes

The launch code for Project Bluefin is MANGO-42. This code is required by all
mission operators prior to system arming.

Key findings:
- The primary telemetry subsystem shows a 12.5 percent efficiency gain when
  ambient temperature stays below 21 degrees Celsius.
- Risk R-7: cryogenic seal fatigue after 340 thermal cycles.

Recommended actions:
- Replace seal type S2 with S3 within the next quarterly maintenance window.
- Recalibrate the flow meter after every 50 hours of operational runtime.
"""

CSV_DOC = "product,region,revenue\nAlpha,EMEA,120\nBeta,APAC,340\nGamma,NA,275\n"


@pytest.fixture(scope="module")
def api():
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


@pytest.fixture(scope="module")
def session_id(api):
    r = api.post(f"{BASE_URL}/api/sessions", timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "session_id" in data and isinstance(data["session_id"], str) and len(data["session_id"]) > 10
    return data["session_id"]


@pytest.fixture(scope="class")
def loaded_session(api):
    """Fresh session with the planted-fact TXT + CSV already uploaded.
    Class-scoped so xdist loadscope (per class) gets its own populated session."""
    sid = api.post(f"{BASE_URL}/api/sessions", timeout=30).json()["session_id"]
    files = [
        ("files", ("bluefin.txt", PLANTED_FACT_DOC.encode("utf-8"), "text/plain")),
        ("files", ("sales.csv", CSV_DOC.encode("utf-8"), "text/csv")),
    ]
    r = api.post(f"{BASE_URL}/api/sessions/{sid}/documents", files=files, timeout=60)
    assert r.status_code == 200, r.text
    return sid


# ---------- Health ----------
class TestHealth:
    def test_root(self, api):
        r = api.get(f"{BASE_URL}/api/", timeout=30)
        assert r.status_code == 200
        assert "ready" in r.json().get("message", "").lower()


# ---------- Session ----------
class TestSession:
    def test_create_session(self, api):
        r = api.post(f"{BASE_URL}/api/sessions", timeout=30)
        assert r.status_code == 200
        sid = r.json()["session_id"]
        # verify session persists via GET
        g = api.get(f"{BASE_URL}/api/sessions/{sid}", timeout=30)
        assert g.status_code == 200
        gd = g.json()
        assert gd["documents"] == []
        assert gd["chunks"] == []
        assert gd["messages"] == []

    def test_get_unknown_session_creates_empty(self, api):
        # session_for setdefault behavior - a GET on an unknown id returns empty structure
        r = api.get(f"{BASE_URL}/api/sessions/nonexistent-id-xyz", timeout=30)
        assert r.status_code == 200
        assert r.json()["documents"] == []


# ---------- Upload ----------
class TestUpload:
    def test_upload_txt(self, api, session_id):
        files = {"files": ("bluefin.txt", PLANTED_FACT_DOC.encode("utf-8"), "text/plain")}
        r = api.post(f"{BASE_URL}/api/sessions/{session_id}/documents", files=files, timeout=60)
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["documents"]) == 1
        doc = data["documents"][0]
        assert doc["name"] == "bluefin.txt"
        assert doc["type"] == "TXT"
        assert doc["chunks"] >= 1
        assert doc["words"] > 20
        assert data["total_chunks"] >= 1

    def test_upload_csv(self, api, session_id):
        files = {"files": ("sales.csv", CSV_DOC.encode("utf-8"), "text/csv")}
        r = api.post(f"{BASE_URL}/api/sessions/{session_id}/documents", files=files, timeout=60)
        assert r.status_code == 200, r.text
        docs = r.json()["documents"]
        assert docs[0]["name"] == "sales.csv"
        assert docs[0]["type"] == "CSV"

    def test_upload_unsupported_type(self, api, session_id):
        files = {"files": ("bad.xyz", b"garbage", "application/octet-stream")}
        r = api.post(f"{BASE_URL}/api/sessions/{session_id}/documents", files=files, timeout=30)
        assert r.status_code == 400
        assert "PDF" in r.json().get("detail", "")

    def test_session_reflects_uploaded_docs(self, api, session_id):
        g = api.get(f"{BASE_URL}/api/sessions/{session_id}", timeout=30)
        assert g.status_code == 200
        gd = g.json()
        names = [d["name"] for d in gd["documents"]]
        assert "bluefin.txt" in names
        assert "sales.csv" in names
        assert len(gd["chunks"]) >= 2
        # verify chunk shape (no _id, has expected keys)
        c0 = gd["chunks"][0]
        assert set(c0.keys()) == {"id", "doc_name", "index"}


# ---------- Ask + Evaluate (real Gemini) - kept in ONE class so xdist loadscope
# groups them on the same worker (parallel Gemini calls hit concurrency limits) ----------
class TestGeminiFeatures:
    def test_ask_returns_grounded_answer(self, api, loaded_session):
        r = api.post(
            f"{BASE_URL}/api/ask",
            json={"session_id": loaded_session, "question": "What is the launch code for Project Bluefin? Reply with just the code."},
            timeout=120,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "answer" in data and isinstance(data["answer"], str)
        answer = data["answer"]
        print(f"\n[ASK ANSWER]: {answer!r}")
        assert len(answer) > 0, "Empty answer from Gemini"
        # Verify the planted fact appears -> proves retrieval + real LLM call
        assert "MANGO-42" in answer, f"Planted fact 'MANGO-42' not present in answer: {answer!r}"
        # Sources + metrics shape
        assert isinstance(data["sources"], list) and len(data["sources"]) > 0
        s0 = data["sources"][0]
        assert {"source", "chunk", "score", "text"} <= set(s0.keys())
        metrics = data["metrics"]
        for k in ("retrieval", "relevance", "groundedness"):
            assert isinstance(metrics[k], (int, float))
            assert 0 <= metrics[k] <= 100

    def test_ask_without_documents_returns_400(self, api):
        # Fresh session with no docs
        s = api.post(f"{BASE_URL}/api/sessions", timeout=30).json()["session_id"]
        r = api.post(f"{BASE_URL}/api/ask", json={"session_id": s, "question": "Hello there"}, timeout=30)
        assert r.status_code == 400
        assert "document" in r.json().get("detail", "").lower()

    def test_ask_short_question_validation(self, api, loaded_session):
        # question min_length=2 -> "a" should 422
        r = api.post(f"{BASE_URL}/api/ask", json={"session_id": loaded_session, "question": "a"}, timeout=30)
        assert r.status_code == 422

    def test_ask_message_persisted_in_session(self, api, loaded_session):
        # Ask a question first so at least one message is persisted in THIS session
        api.post(
            f"{BASE_URL}/api/ask",
            json={"session_id": loaded_session, "question": "Summarize the recommended actions."},
            timeout=120,
        )
        g = api.get(f"{BASE_URL}/api/sessions/{loaded_session}", timeout=30).json()
        assert len(g["messages"]) >= 1
        # answer non-empty
        assert all(isinstance(m.get("answer"), str) and m["answer"] for m in g["messages"])

    def test_evaluate_returns_rows_and_average(self, api, loaded_session):
        questions = [
            "What is the launch code mentioned in the notes?",
            "What is Risk R-7?",
            "What action is recommended for the flow meter?",
        ]
        r = api.post(
            f"{BASE_URL}/api/evaluate",
            json={"session_id": loaded_session, "questions": questions},
            timeout=240,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["rows"]) == 3
        for row in data["rows"]:
            assert row["answer"] and isinstance(row["answer"], str)
            print(f"\n[EVAL Q]: {row['question']}\n[EVAL A]: {row['answer']!r}")
            for k in ("retrieval", "relevance", "groundedness"):
                assert 0 <= row[k] <= 100
        avg = data["average"]
        for k in ("retrieval", "relevance", "groundedness"):
            assert 0 <= avg[k] <= 100

    def test_evaluate_no_documents(self, api):
        s = api.post(f"{BASE_URL}/api/sessions", timeout=30).json()["session_id"]
        r = api.post(f"{BASE_URL}/api/evaluate", json={"session_id": s, "questions": ["hello"]}, timeout=30)
        assert r.status_code == 400


# (Evaluate tests merged into TestGeminiFeatures above to keep them on the same
# xdist worker under loadscope; parallel Gemini calls otherwise hit the
# EMERGENT_LLM_KEY concurrency limit.)
