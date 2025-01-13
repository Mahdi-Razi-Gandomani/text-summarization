import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, BartTokenizer
from kneed import KneeLocator

# Example text
text = """
Python is one of the most popular programming languages in the world today. It is known for its readability and simplicity, making it an excellent choice for beginners. Python was developed by Guido van Rossum and first released in 1991. Over the years, it has become a dominant language in fields like web development, data science, machine learning, and automation.

Python’s syntax is easy to learn and intuitive, which allows programmers to write less code to accomplish more. This makes it great for rapid prototyping and development. Python's simplicity also translates into a large and active community. There are many resources available online, including tutorials, documentation, and forums where people can seek help.

In addition to its ease of use, Python has a rich ecosystem of libraries and frameworks. Libraries like NumPy and pandas make data manipulation and analysis straightforward. For machine learning, TensorFlow, PyTorch, and scikit-learn are widely used libraries that help developers build complex models with ease. Django and Flask are two popular web frameworks that make it easy to develop web applications quickly.

Python is highly versatile, running on various platforms, including Windows, macOS, and Linux. It also has a number of powerful tools for automating tasks, like web scraping with BeautifulSoup or Selenium, or task scheduling with Celery. Python’s versatility has made it a go-to tool for many software developers, analysts, and engineers.

Despite its popularity, Python is not without its limitations. It is slower than compiled languages like C++ and Java, making it less ideal for applications that require extreme performance. However, Python can still be used effectively in many high-performance applications by integrating with faster languages or using optimizations like NumPy’s array processing.

The future of Python looks bright. It continues to grow in popularity due to its active development and large community. Python is constantly evolving, with regular updates and new features being added to make it even more powerful and user-friendly. As more industries adopt data science and machine learning, Python is expected to remain a leading language in the tech industry for years to come.
"""

sentences = nltk.sent_tokenize(text)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get embeddings
sentence_embeddings = model.encode(sentences)

# Determine the optimal number of clusters using KneeLocator
inertia = []
cluster_range = range(1, 10)  # Test cluster numbers from 1 to 9
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(sentence_embeddings)
    inertia.append(kmeans.inertia_)
knee_locator = KneeLocator(cluster_range, inertia, curve="convex", direction="decreasing")
optimal_clusters = knee_locator.knee

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(sentence_embeddings)
centroids = kmeans.cluster_centers_

# Construct a query sentence from the first three sentences of the article
query_sentence = ' '.join(sentences[:3])
query_vector = model.encode([query_sentence])

# Function to find the closest sentence from each cluster to query
def find_summary_sentence(cluster_idx, sentence_embeddings, sentences):
    cluster_sentences_idx = np.where(kmeans.labels_ == cluster_idx)[0]
    cluster_embeddings = sentence_embeddings[cluster_sentences_idx]
    similarities = cosine_similarity(cluster_embeddings, query_vector)
    most_similar_idx = cluster_sentences_idx[np.argmax(similarities)]
    return sentences[most_similar_idx]

# Find a summary sentence for each cluster
summary_sentences = []
for cluster_idx in range(optimal_clusters):
    summary_sentence = find_summary_sentence(cluster_idx, centroids, sentence_embeddings, sentences)
    summary_sentences.append(summary_sentence)
extractive_summary = " ".join(summary_sentences)

# Load pre-trained BART model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
inputs = tokenizer(extractive_summary, return_tensors="pt", max_length=1024, truncation=True)

# Generate Final Summary
summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Display the summary
print("\nAbstractive Summary:")
print(final_summary)
