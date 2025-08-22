# Semantic Book Recommender

This project implements a **semantic-based book recommendation system** using Natural Language Processing (NLP) and embeddings.   
Instead of relying only on metadata or ratings, the system understands the semantic meaning of book descriptions and recommends similar books based on context.  

## Overview

- **Dataset Processing:**  
  Loads and processes book descriptions or summaries to prepare for embedding generation.

- **Semantic Embeddings:**  
  Converts book descriptions into vector embeddings to capture the semantic meaning of text.

- **Similarity Search:**  
  Uses cosine similarity to compare embeddings and recommend books with the closest semantic match.

- **Recommendation System:**  
  Given a book title or description, the system suggests other books that are semantically similar.

## Project Structure  
.  
├── README.md # Project documentation  
├── semantic_book_recommender.py # Main script for generating recommendations  
├── books.csv # Dataset of books with descriptions  
└── requirements.txt # Python dependencies  


## Requirements
- Python 3.x  
- Libraries:  
  - pandas  
  - numpy  
  - scikit-learn  
  - sentence-transformers (for embeddings)  
