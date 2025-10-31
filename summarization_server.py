from flask import Flask, request, jsonify
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from dotenv import load_dotenv


load_dotenv()

nltk.download('punkt')

app = Flask(__name__)

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Hugging Face API
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
BART_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"


# Call Hugging Face API for using BART.
def abstractive_summary_api(text, max_length=1024, min_length=64):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,      
            "min_length": min_length,      
            "do_sample": True,             
            "early_stopping": False
        }
    }

    try:
        response = requests.post(BART_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result[0]['summary_text']
    except Exception:
        raise RuntimeError("Hugging Face API request failed")


def hybrid_summarize(text, max_length=1024, min_length=64):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 3:
        return "Text too short to summarize."

    # Get embeddings
    sentence_embeddings = sentence_model.encode(sentences)

    # Find optimal number of clusters
    cluster_range = range(2, len(sentences) // 2)
    silhouette_scores = []
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(sentence_embeddings)
        score = silhouette_score(sentence_embeddings, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]

    # KMeans clustering
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(sentence_embeddings)

    # Build a query from first few sentences
    query_sentence = ' '.join(sentences[:3])
    query_vector = sentence_model.encode([query_sentence])

    # Pick representative sentences
    summary_sentences = []
    for cluster_idx in range(optimal_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_idx)[0]
        cluster_embeddings = sentence_embeddings[cluster_indices]
        similarities = cosine_similarity(cluster_embeddings, query_vector)
        best_idx = cluster_indices[np.argmax(similarities)]
        summary_sentences.append(sentences[best_idx])

    extractive_summary = " ".join(summary_sentences)

    # Get final summary
    abstractive_summary = abstractive_summary_api(extractive_summary, max_length, min_length)

    return abstractive_summary


# summarize endpoint
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required field: text"}), 400

        text = data['text']
        max_length = data.get('max_length', 1024)
        min_length = data.get('min_length', 64)

        summary = hybrid_summarize(text, max_length, min_length)

        return jsonify({
            "summary": summary,
            "metadata": {
                "original_length": len(text),
                "summary_length": len(summary),
                "max_length": max_length,
                "min_length": min_length
            }
        }), 200

    except PermissionError as e:
        return jsonify({"error": str(e)}), 401
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# root endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Hybrid Text Summarization API",
        "endpoints": {
            "/summarize": "POST Generate summary",
        },
        "usage": {
            "method": "POST",
            "url": "/summarize",
            "payload": {
                "text": "Your text",
                "max_length": 1024,
                "min_length": 64
            }
        }
    })


if __name__ == '__main__':
    app.run()
