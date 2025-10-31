# Hybrid Text Summarization API

A Flask-based REST API that combines extractive and abstractive summarization techniques to generate high-quality text summaries. The system uses sentence embeddings, K-means clustering, and the BART model to produce concise, coherent summaries.

## Features

1. **Extractive Phase**: 
   - Tokenizes text into sentences
   - Generates sentence embeddings
   - Clusters sentences using K-means with optimal cluster selection
   - Selects representative sentences from each cluster

2. **Abstractive Phase**:
   - Feeds extractive summary to BART model via Hugging Face API
   - Generates final polished summary

---

## Project Structure

```
project/
│
├─ summarization_server.py           # Flask server with summarization endpoint
├─ summarization_client.py           # Example client for testing the API
├─ requirements.txt                  # Python package dependencies
└─ README.md
```
---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mahdi-Razi-Gandomani/text-summarization.git
   cd text-summarization

2. Install Dependencies

  ```bash
  pip install -r requirements.txt
  ```

3. Start the Server

  ```bash
  python summarization_server.py
  ```

The server will start on `http://127.0.0.1:5000`

---

## API Endpoints

#### API Information
**Endpoint**: `GET /`

Returns information about available endpoints and usage.

**Response:**
```json
{
  "message": "Hybrid Text Summarization API",
  "endpoints": {
    "/summarize": "POST Generate summary"
  }
}
```

#### Summarize Text
**Endpoint**: `POST /summarize`

Generate a summary for provided text

**Request:**
```json
{
  "text": "Your text here...",
  "max_length": 1024,
  "min_length": 64
}
```

**Parameters:**
- `text` (required): The text to summarize
- `max_length` (optional, default: 1024): Maximum summary length
- `min_length` (optional, default: 64): Minimum summary length

**Response:**
```json
{
  "summary": "Generated summary text...",
  "metadata": {
    "original_length": 1523,
    "summary_length": 234,
    "max_length": 1024,
    "min_length": 64
  }
}
```

### Example Client Usage

Replace your text and run the provided client:

```bash
python summarization_client.py
```

Or use curl:

```bash
curl -X POST http://127.0.0.1:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "max_length": 1024,
    "min_length": 64
  }'
```

---

## References

- **Optimizing Text Summarization**: Edrees, Z., & Ortakci, Y. (2024). *Optimizing Text Summarization with Sentence Clustering and Natural Language Processing*. International Journal of Advanced Computer Science and Applications, 15(10), 1123-1132. DOI: [10.14569/IJACSA.2024.01510115](https://doi.org/10.14569/IJACSA.2024.01510115).
