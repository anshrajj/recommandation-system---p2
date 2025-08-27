import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self):
        self.user_movie_ratings, self.movies = self.create_sample_data()
        self.user_similarity = None
        self.calculate_similarity()
    
    def create_sample_data(self):
        """Create sample user-movie rating data"""
        # Sample movie titles
        movies = [
            'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
            'Pulp Fiction', 'Inception', 'Interstellar', 'Parasite',
            'The Matrix', 'Goodfellas', 'The Departed'
        ]
        
        # Sample user ratings (0 means not rated)
        user_ratings = {
            'User1': [5, 4, 0, 0, 5, 4, 0, 5, 0, 3],
            'User2': [0, 5, 4, 3, 0, 0, 5, 4, 5, 0],
            'User3': [4, 0, 5, 0, 4, 5, 0, 0, 4, 5],
            'User4': [3, 4, 0, 5, 0, 0, 4, 3, 0, 4],
            'User5': [0, 0, 4, 0, 5, 4, 5, 0, 3, 0],
            'Current_User': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Will be updated
        }
        
        return pd.DataFrame(user_ratings, index=movies), movies
    
    def calculate_similarity(self):
        """Calculate user similarity matrix"""
        # Replace 0 with NaN for better similarity calculation
        ratings = self.user_movie_ratings.replace(0, np.nan)
        self.user_similarity = cosine_similarity(ratings.fillna(0).T)
    
    def get_user_ratings(self):
        """Get ratings from the current user"""
        print("Welcome to Collaborative Filtering Recommendation System!")
        print("\nPlease rate the following movies (1-5, 0 if not seen):")
        
        for i, movie in enumerate(self.movies, 1):
            while True:
                try:
                    rating = int(input(f"{i}. {movie}: "))
                    if 0 <= rating <= 5:
                        self.user_movie_ratings.loc[movie, 'Current_User'] = rating
                        break
                    else:
                        print("Please enter a rating between 0 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
    
    def recommend_movies(self, top_n=3):
        """Recommend movies using collaborative filtering"""
        # Get current user's ratings
        current_user_ratings = self.user_movie_ratings['Current_User']
        
        # Find similar users (excluding current user)
        user_names = self.user_movie_ratings.columns.tolist()
        current_user_idx = user_names.index('Current_User')
        
        # Get similarity scores with other users
        similarities = self.user_similarity[current_user_idx]
        
        # Exclude similarity with self
        similarities[current_user_idx] = 0
        
        # Get top similar users
        similar_users_idx = similarities.argsort()[-3:][::-1]
        similar_users = [user_names[i] for i in similar_users_idx]
        
        # Predict ratings for unrated movies
        predictions = {}
        for movie in self.movies:
            if current_user_ratings[movie] == 0:  # Movie not rated by current user
                weighted_sum = 0
                similarity_sum = 0
                
                for user in similar_users:
                    rating = self.user_movie_ratings.loc[movie, user]
                    if rating > 0:  # If the similar user rated this movie
                        user_idx = user_names.index(user)
                        similarity = similarities[user_idx]
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    predictions[movie] = predicted_rating
        
        # Get top recommendations
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return recommendations
    
    def display_recommendations(self, recommendations):
        """Display the recommendations"""
        print(f"\n{'='*60}")
        print("TOP MOVIE RECOMMENDATIONS (Collaborative Filtering)")
        print(f"{'='*60}")
        
        for i, (movie, score) in enumerate(recommendations, 1):
            print(f"{i}. {movie}")
            print(f"   Predicted Rating: {score:.2f}/5")
            print()

def main():
    recommender = CollaborativeRecommender()
    recommender.get_user_ratings()
    recommendations = recommender.recommend_movies()
    recommender.display_recommendations(recommendations)

if __name__ == "__main__":
    main()
