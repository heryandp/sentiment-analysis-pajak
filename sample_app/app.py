from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT_DIR / "sentiment_model_final.pkl"))
VECTORIZER_PATH = Path(os.environ.get("VECTORIZER_PATH", ROOT_DIR / "tfidf_vectorizer.pkl"))

app = Flask(__name__)

MODEL = None
VECTORIZER = None
MODEL_ERROR = None


def _load_assets() -> None:
    global MODEL, VECTORIZER, MODEL_ERROR
    if joblib is None:
        MODEL_ERROR = "joblib is not installed. Install dependencies first."
        return
    try:
        VECTORIZER = joblib.load(VECTORIZER_PATH)
        MODEL = joblib.load(MODEL_PATH)
        MODEL_ERROR = None
    except Exception as exc:  # pragma: no cover
        MODEL = None
        VECTORIZER = None
        MODEL_ERROR = f"Failed to load model assets: {exc}"
        return



def _basic_preprocess(text: str) -> str:
    return " ".join(text.strip().split())


def _predict(text: str) -> Dict[str, Any]:
    if MODEL is None or VECTORIZER is None:
        raise RuntimeError(MODEL_ERROR or "Model not loaded")

    cleaned = _basic_preprocess(text)
    X = VECTORIZER.transform([cleaned])
    pred = MODEL.predict(X)[0]

    debug: Dict[str, Any] = {
        "input": text,
        "input_length": len(text),
        "cleaned_length": len(cleaned),
        "vector_shape": list(X.shape),
        "model_class": MODEL.__class__.__name__,
        "vectorizer_class": VECTORIZER.__class__.__name__,
    }

    # Best-effort score/proba
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)[0]
        classes = list(getattr(MODEL, "classes_", []))
        debug["probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}
    elif hasattr(MODEL, "decision_function"):
        scores = MODEL.decision_function(X)
        try:
            scores = scores[0]
        except Exception:
            pass
        debug["decision_scores"] = scores.tolist() if hasattr(scores, "tolist") else scores

    return {
        "sentiment": str(pred),
        "debug": debug,
    }


def _predict_many(texts: list[str]) -> Dict[str, Any]:
    cleaned_list = [_basic_preprocess(t) for t in texts]
    X = VECTORIZER.transform(cleaned_list)
    preds = MODEL.predict(X)

    debug: Dict[str, Any] = {
        "inputs": texts,
        "count": len(texts),
        "vector_shape": list(X.shape),
        "model_class": MODEL.__class__.__name__,
        "vectorizer_class": VECTORIZER.__class__.__name__,
    }

    results = []
    for raw, cleaned, pred in zip(texts, cleaned_list, preds):
        item = {
            "text": raw,
            "sentiment": str(pred),
            "debug": {
                "input_length": len(raw),
                "cleaned_length": len(cleaned),
            },
        }
        results.append(item)

    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)
        classes = list(getattr(MODEL, "classes_", []))
        for item, row in zip(results, probs):
            item["debug"]["probabilities"] = {
                str(c): float(p) for c, p in zip(classes, row)
            }
    elif hasattr(MODEL, "decision_function"):
        scores = MODEL.decision_function(X)
        try:
            scores = scores.tolist()
        except Exception:
            pass
        for item, row in zip(results, scores):
            item["debug"]["decision_scores"] = row

    return {
        "results": results,
        "debug": debug,
    }


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sentiment Demo</title>
  <style>
    :root {
      --bg: #f7f4f0;
      --fg: #1d1b1a;
      --muted: #6a6661;
      --accent: #1b6e5a;
      --card: #ffffff;
      --border: #e2ddd7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      color: var(--fg);
      background: radial-gradient(1200px 400px at 10% 0%, #efe9e1, transparent), var(--bg);
    }
    .wrap {
      max-width: 960px;
      margin: 32px auto;
      padding: 0 16px;
    }
    header {
      display: flex;
      gap: 16px;
      align-items: center;
      margin-bottom: 20px;
    }
    h1 { margin: 0; font-size: 28px; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }
    textarea {
      width: 100%;
      min-height: 160px;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      font-size: 15px;
      resize: vertical;
    }
    .row { display: flex; gap: 12px; margin-top: 12px; }
    button {
      background: var(--accent);
      color: #fff;
      border: 0;
      border-radius: 10px;
      padding: 10px 16px;
      font-weight: 600;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .result {
      margin-top: 16px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 12px;
      border-radius: 10px;
      overflow: auto;
      font-size: 13px;
    }
    .badge {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e7f3ef;
      color: #145544;
      font-weight: 700;
    }
    .muted { color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Sentiment Analysis Demo</h1>
        <div class="muted">Type a sentence, get sentiment + debug.</div>
      </div>
    </header>

    <div class="card">
      <div id="inputs"></div>
      <div class="row">
        <button id="addInput" type="button">Add Input</button>
        <button id="run">Analyze</button>
        <button id="sample" type="button">Sample</button>
        <button id="sampleMulti" type="button">Sample Multi</button>
      </div>
      <div class="result" id="result" style="display:none;">
        <div>Sentiment: <span class="badge" id="sentiment"></span></div>
        <div>
          <div class="muted">Debug</div>
          <pre id="debug"></pre>
        </div>
      </div>
    </div>
  </div>

<script>
const runBtn = document.getElementById('run');
const sampleBtn = document.getElementById('sample');
const sampleMultiBtn = document.getElementById('sampleMulti');
const addInputBtn = document.getElementById('addInput');
const resultBox = document.getElementById('result');
const sentimentEl = document.getElementById('sentiment');
const debugEl = document.getElementById('debug');
const inputsEl = document.getElementById('inputs');

function addInput(value = "") {
  const wrap = document.createElement('div');
  wrap.style.marginTop = '8px';
  const ta = document.createElement('textarea');
  ta.placeholder = "Masukkan teks di sini...";
  ta.value = value;
  ta.style.minHeight = '120px';
  ta.style.width = '100%';
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.textContent = 'Remove';
  remove.style.marginTop = '6px';
  remove.onclick = () => wrap.remove();
  wrap.appendChild(ta);
  wrap.appendChild(remove);
  inputsEl.appendChild(wrap);
}

addInput();

sampleBtn.onclick = () => {
  inputsEl.innerHTML = '';
  addInput("Kebijakan pajak baru ini sangat membantu UMKM berkembang.");
};
sampleMultiBtn.onclick = () => {
  inputsEl.innerHTML = '';
  addInput("Kebijakan pajak baru ini sangat membantu UMKM berkembang.");
  addInput("Layanan pajak online masih sering bermasalah dan membuat frustrasi.");
  addInput("Informasi tentang aturan pajak perlu disosialisasikan lebih jelas.");
};

addInputBtn.onclick = () => addInput();

runBtn.onclick = async () => {
  const values = Array.from(inputsEl.querySelectorAll('textarea'))
    .map(t => t.value.trim())
    .filter(Boolean);
  if (values.length === 0) return;
  runBtn.disabled = true;
  try {
    const body = values.length > 1 ? { texts: values } : { text: values[0] };
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    if (data.results) {
      sentimentEl.textContent = data.results.map(r => r.sentiment).join(", ");
      debugEl.textContent = JSON.stringify(data, null, 2);
    } else {
      sentimentEl.textContent = data.sentiment;
      debugEl.textContent = JSON.stringify(data.debug, null, 2);
    }
    resultBox.style.display = 'grid';
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    runBtn.disabled = false;
  }
};
</script>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    return render_template_string(INDEX_HTML)


@app.route("/health")
def health() -> Any:
    return jsonify(
        {
            "ok": MODEL is not None and VECTORIZER is not None,
            "model_path": str(MODEL_PATH),
            "vectorizer_path": str(VECTORIZER_PATH),
            "error": MODEL_ERROR,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict() -> Any:
    if not request.is_json:
        return jsonify({"error": "Expected JSON body"}), 400
    payload = request.get_json(force=True) or {}
    text = payload.get("text", "")
    texts = payload.get("texts", None)
    if texts is not None:
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "`texts` must be a non-empty list"}), 400
        normalized = [str(t) for t in texts if str(t).strip()]
        if not normalized:
            return jsonify({"error": "All texts are empty"}), 400
        try:
            result = _predict_many(normalized)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    text = str(text)
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400
    try:
        result = _predict(text)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    _load_assets()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
