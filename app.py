from __future__ import annotations

import json
import os
import re
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "avis_clients.csv"
HOST = os.getenv("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

STOP_WORDS = {
    "a", "afin", "ai", "aie", "aient", "aies", "ait", "alors", "apres", "as",
    "au", "aucun", "aussi", "autre", "avant", "avec", "avoir", "bon", "car",
    "ce", "cela", "ces", "cet", "cette", "chez", "comme", "dans", "de",
    "des", "du", "elle", "elles", "en", "encore", "est", "et", "etre",
    "eu", "fait", "font", "ici", "il", "ils", "je", "la", "le", "les",
    "leur", "lui", "ma", "mais", "me", "mes", "moi", "mon", "ne", "nos",
    "notre", "nous", "on", "ont", "ou", "par", "pas", "peu", "plus",
    "pour", "que", "qui", "sa", "se", "ses", "son", "sont", "sur", "ta",
    "te", "tes", "toi", "ton", "tous", "tout", "tres", "tu", "un", "une",
    "vos", "votre", "vous", "y",
}

POSITIVE_WORDS = {
    "agreable", "attentif", "avance", "bon", "bonne", "captivant", "clair",
    "claire", "conforme", "content", "delicieux", "efficace", "excellent",
    "excellente", "facile", "fluide", "impeccable", "intuitive", "parfaitement",
    "professionnelle", "propre", "qualite", "rapide", "reactive", "recommande",
    "robuste", "satisfait", "simple", "superieure", "utile",
}

NEUTRAL_WORDS = {
    "acceptable", "assez", "convenable", "correct", "correcte", "delais",
    "fonctionnelle", "manque", "moyen", "moyenne", "normal", "normale",
    "particulier", "peut", "pourrait", "standard",
}

NEGATIVE_WORDS = {
    "abime", "bugs", "bruyant", "casse", "complique", "confuse", "decu",
    "decevant", "defectueux", "desagreable", "difficile", "ennuyeux", "erreurs",
    "faible", "fragile", "froid", "incomplete", "instable", "inutile", "lente",
    "longue", "mauvaise", "mediocre", "panne", "probleme", "retour", "sale",
    "tard", "trop",
}


class LexiconFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        rows = []
        for text in texts:
            tokens = normalize_text(text).split()
            total = max(len(tokens), 1)
            rows.append(
                [
                    sum(token in POSITIVE_WORDS for token in tokens) / total,
                    sum(token in NEUTRAL_WORDS for token in tokens) / total,
                    sum(token in NEGATIVE_WORDS for token in tokens) / total,
                ]
            )
        return csr_matrix(rows)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s']", " ", text)
    tokens = [token for token in text.split() if token not in STOP_WORDS and len(token) > 2]
    return " ".join(tokens)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("neutre").astype(str)
    df["clean_text"] = df["text"].apply(normalize_text)
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df


def build_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=df["sentiment"],
    )
    model = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "lexicon",
                            LexiconFeatures(),
                        ),
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                preprocessor=normalize_text,
                                ngram_range=(1, 2),
                                min_df=1,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                preprocessor=normalize_text,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(df["sentiment"].unique())
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "test_size": len(x_test),
        "train_size": len(x_train),
    }
    return model, metrics


DATA = load_data()
MODEL, MODEL_METRICS = build_model(DATA)


def top_words(df: pd.DataFrame, sentiment: str | None = None, limit: int = 12) -> list[dict]:
    subset = df if sentiment in (None, "all") else df[df["sentiment"] == sentiment]
    words = " ".join(subset["clean_text"]).split()
    return [{"word": word, "count": count} for word, count in Counter(words).most_common(limit)]


def dataset_records(limit: int = 100) -> list[dict]:
    columns = ["id", "text", "sentiment", "source", "word_count"]
    return DATA[columns].head(limit).to_dict(orient="records")


def summary_payload() -> dict:
    distribution = DATA["sentiment"].value_counts().to_dict()
    source_distribution = DATA["source"].value_counts().to_dict()
    length_by_sentiment = (
        DATA.groupby("sentiment")["word_count"].mean().round(1).sort_index().to_dict()
    )
    return {
        "rows": int(len(DATA)),
        "columns": list(DATA.columns),
        "missing_values": DATA[["text", "sentiment"]].isna().sum().to_dict(),
        "sentiment_distribution": distribution,
        "source_distribution": source_distribution,
        "avg_words": round(float(DATA["word_count"].mean()), 1),
        "avg_chars": round(float(DATA["text_length"].mean()), 1),
        "length_by_sentiment": length_by_sentiment,
        "top_words": top_words(DATA),
        "top_words_by_sentiment": {
            sentiment: top_words(DATA, sentiment) for sentiment in sorted(DATA["sentiment"].unique())
        },
    }


def predict_payload(text: str) -> dict:
    cleaned = normalize_text(text)
    prediction = str(MODEL.predict([text])[0])
    probabilities = MODEL.predict_proba([text])[0]
    classes = MODEL.classes_
    scores = {
        label: round(float(score), 4)
        for label, score in sorted(zip(classes, probabilities), key=lambda item: item[0])
    }
    return {
        "text": text,
        "clean_text": cleaned,
        "sentiment": prediction,
        "confidence": round(float(max(probabilities)), 4),
        "scores": scores,
    }


class AppHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(ROOT / "templates" / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/summary":
            self.send_json(summary_payload())
        elif parsed.path == "/api/dataset":
            self.send_json({"records": dataset_records()})
        elif parsed.path == "/api/metrics":
            self.send_json(MODEL_METRICS)
        elif parsed.path.startswith("/static/"):
            requested = (ROOT / parsed.path.lstrip("/")).resolve()
            if ROOT.resolve() not in requested.parents:
                self.send_error(403)
                return
            content_type = "text/css" if requested.suffix == ".css" else "application/javascript"
            self.serve_file(requested, content_type)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
            text = str(payload.get("text", "")).strip()
            if not text:
                self.send_json({"error": "Le texte est obligatoire."}, status=400)
                return
            self.send_json(predict_payload(text))
        except json.JSONDecodeError:
            self.send_json({"error": "JSON invalide."}, status=400)

    def serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict, status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    display_host = "127.0.0.1" if HOST == "0.0.0.0" else HOST
    print(f"Application lancee sur http://{display_host}:{PORT}")
    print("Appuie sur Ctrl+C pour arreter le serveur.")
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
