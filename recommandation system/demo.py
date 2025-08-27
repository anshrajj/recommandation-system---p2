#!/usr/bin/env python3
"""
Demo script showing the recommendation system logic
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def demo_content_based():
    """Demonstrate content-based filtering"""
    print("=== CONTENT-BASED RECOMMENDATION DEMO ===\n")
    
    # Sample movie data
    movies = pd.DataFrame({
        'title': [
            'The Matrix', 'Inception', 'The Shawshank Redemption',
            'The Godfather', 'Pulp Fiction', 'Interstellar'
        ],
        'genres': [
            'Action Sci-Fi', 'Action Sci-Fi Thriller', 'Drama Crime',
            'Crime Drama', 'Crime Drama', 'Adventure Drama Sci-Fi'
        ]
    })
    
    print("Movie Dataset:")
    for i, row in movies.iterrows():
        print(f"{i+1}. {row['title']} - {row['genres']}")
    
    # Vectorize genres
    vectorizer = TfidfVectorizer(stop_words='english')
    genre_matrix = vectorizer.fit_transform(movies['genres'])
    
    print(f"\nVectorized genres shape: {genre_matrix.shape}")
    print("Feature names:", vectorizer.get_feature_names_out())
    
    # User preferences
    user_prefs = "Action Sci-Fi"
    print(f"\nUser preferences: {user_prefs}")
    
    # Vectorize user preferences
    user_vector = vectorizer.transform([user_prefs])
    
    # Calculate similarity
    similarity_scores = cosine_similarity(user_vector, genre_matrix).flatten()
    
    print("\nSimilarity scores:")
    for i, (title, score) in enumerate(zip(movies['title'], similarity_scores)):
        print(f"{title}: {score:.3f}")
    
    # Get recommendations
    top_indices = similarity_scores.argsort()[-3:][::-1]
    print("\nTop Recommendations:")
    for idx in top_indices:
        movie = movies.iloc[idx]
        print(f"- {movie['title']} (score: {similarity_scores[idx]:.3f})")

def demo_collaborative():
    """Demonstrate collaborative filtering concepts"""
    print("\n=== COLLABORATIVE FILTERING CONCEPTS ===\n")
    
    # Sample user-item matrix
    ratings = pd.DataFrame({
        'User1': [5, 4, 0, 0, 3],
        'User2': [0, 5, 4, 3, 0],
        'User3': [4, 0, 5, 0, 4],
        'Current_User': [0, 0, 0, 0, 0]
    }, index=['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5'])
    
    print("User-Item Rating Matrix:")
    print(ratings)
    
    # Calculate similarity
    similarity = cosine_similarity(ratings.fillna(0).T)
    print(f"\nUser Similarity Matrix:\n{similarity}")
    
    print("\nThis demonstrates how users with similar rating patterns")
    print("can be used to predict ratings for unseen movies!")

if __name__ == "__main__":
    demo_content_based()
    demo_collaborative()
    print("\n=== DEMO COMPLETE ===")
    print("\nTo run the full interactive systems:")
    print("1. Content-based: python app.py")
    print("2. Collaborative: python collaborative_filtering.py")
