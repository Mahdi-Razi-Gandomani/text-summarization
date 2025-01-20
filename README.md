# Text Summarization with Extractive and Abstractive Methods

This project demonstrates a hybrid approach to text summarization, combining **extractive** and **abstractive** methods. It uses clustering to extract key sentences and a pre-trained BART model to generate a concise abstractive summary.

---

## Features

- **Extractive Summarization**:
  - Uses clustering to group similar sentences.
  - Selects the most representative sentence from each cluster based on similarity to a query.

- **Abstractive Summarization**:
  - Uses a pre-trained BART model to generate a concise summary from the extracted sentences.

- **Optimal Cluster Selection**:
  - Determines the optimal number of clusters using the **Elbow Method** (KneeLocator).

---

## Requirements

To run this code, you need the following Python libraries:

- `numpy`
- `nltk`
- `sentence-transformers`
- `scikit-learn`
- `transformers`
- `kneed`

---

## Code Structure

### 1. Text Preprocessing
- The input text is split into sentences using `nltk.sent_tokenize`.

### 2. Sentence Embeddings
- Sentence embeddings are generated using the `all-MiniLM-L6-v2` model from `sentence-transformers`.

### 3. Clustering
- **KMeans Clustering** is applied to group similar sentences.
- The optimal number of clusters is determined using the **Elbow Method** (`KneeLocator`).

### 4. Extractive Summarization
- A query sentence is constructed from the first few sentences of the text.
- The most representative sentence from each cluster is selected based on cosine similarity to the query.

### 5. Abstractive Summarization
- The extracted sentences are passed to a pre-trained BART model (`facebook/bart-large-cnn`) to generate a concise summary.

---

## Usage

1. Replace the `text` variable with your input text.
2. Run the script:

   ```bash
   python text_summarization.py

---

## References

- **Optimizing Text Summarization**: Edrees, Z., & Ortakci, Y. (2024). *Optimizing Text Summarization with Sentence Clustering and Natural Language Processing*. International Journal of Advanced Computer Science and Applications, 15(10), 1123-1132. DOI: [10.14569/IJACSA.2024.01510115](https://doi.org/10.14569/IJACSA.2024.01510115).
