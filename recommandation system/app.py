import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MovieRecommender:
    def __init__(self):
        self.movies = self.create_sample_data()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.genre_matrix = self.vectorizer.fit_transform(self.movies['genres'])
        
    def create_sample_data(self):
        """Create a sample dataset of movies with genres"""
        data = {
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
                'Pulp Fiction', 'Fight Club', 'Inception', 'Goodfellas',
                'The Matrix', 'Seven', 'Interstellar', 'Parasite', 'Joker',
                'Whiplash', 'The Departed', 'The Prestige', 'Memento',
                'Gladiator', 'The Lion King', 'Alien', 'The Terminator',
                'Back to the Future', 'Raiders of the Lost Ark', 'Die Hard',
                'The Silence of the Lambs', 'Saving Private Ryan', 'The Green Mile',
                'Forrest Gump', 'Schindler\'s List', 'The Usual Suspects', 'Se7en',
                'The Sixth Sense', 'American Beauty', 'The Shining', 'Braveheart',
                'Good Will Hunting', 'Toy Story', 'Finding Nemo', 'Up',
                'The Incredibles', 'Ratatouille', 'The Dark Knight Rises',
                'Avengers: Infinity War', 'Black Panther', 'Get Out',
                'Mad Max: Fury Road', 'La La Land', 'Moonlight', 'Arrival',
                'Baby Driver', 'Dunkirk'
            ],
            'genres': [
                'Drama Crime', 'Crime Drama', 'Action Crime Drama Thriller',
                'Crime Drama', 'Drama', 'Action Adventure Sci-Fi Thriller',
                'Biography Crime Drama', 'Action Sci-Fi', 'Crime Drama Mystery Thriller',
                'Adventure Drama Sci-Fi', 'Comedy Drama Thriller', 'Crime Drama Thriller',
                'Drama Music', 'Crime Drama Thriller', 'Drama Mystery Thriller',
                'Mystery Thriller', 'Action Adventure Drama', 'Animation Adventure Drama',
                'Horror Sci-Fi', 'Action Sci-Fi', 'Adventure Comedy Sci-Fi',
                'Action Adventure', 'Action Thriller', 'Crime Drama Thriller',
                'Drama War', 'Crime Drama Fantasy', 'Drama Romance',
                'Biography Drama History', 'Crime Mystery Thriller', 'Crime Drama Mystery Thriller',
                'Drama Mystery Thriller', 'Drama', 'Drama Horror', 'Biography Drama History',
                'Drama Romance', 'Animation Adventure Comedy', 'Animation Adventure Comedy',
                'Animation Adventure Comedy', 'Animation Action Adventure', 'Animation Comedy Family',
                'Action Thriller', 'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi',
                'Horror Mystery Thriller', 'Action Adventure Sci-Fi Thriller',
                'Comedy Drama Music Romance', 'Drama', 'Drama Mystery Sci-Fi',
                'Action Crime Drama', 'Action Drama History Thriller'
            ]
        }
        return pd.DataFrame(data)
    
    def get_user_preferences(self):
        """Get user preferences for genres"""
        print("Welcome to the Movie Recommendation System!")
        print("\nAvailable genres: Action, Adventure, Comedy, Crime, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, Animation, Biography, History, Music, War, Family")
        
        preferences = input("\nEnter your preferred genres (comma-separated, e.g., Action, Drama, Sci-Fi): ")
        return preferences.strip()
    
    def recommend_movies(self, user_preferences, top_n=5):
        """Recommend movies based on user preferences"""
        # Vectorize user preferences
        user_vector = self.vectorizer.transform([user_preferences])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, self.genre_matrix).flatten()
        
        # Get top N recommendations
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            movie = self.movies.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity_score': similarity_scores[idx]
            })
        
        return recommendations
    
    def display_recommendations(self, recommendations):
        """Display the recommendations in a formatted way"""
        print(f"\n{'='*60}")
        print("TOP MOVIE RECOMMENDATIONS")
        print(f"{'='*60}")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Genres: {rec['genres']}")
            print(f"   Match Score: {rec['similarity_score']:.3f}")
            print()

def main():
    recommender = MovieRecommender()
    
    while True:
        try:
            user_prefs = recommender.get_user_preferences()
            
            if not user_prefs:
                print("Please enter at least one genre.")
                continue
                
            recommendations = recommender.recommend_movies(user_prefs)
            recommender.display_recommendations(recommendations)
            
            another = input("Would you like to get more recommendations? (yes/no): ").lower()
            if another != 'yes':
                print("Thank you for using the Movie Recommendation System!")
                break
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
