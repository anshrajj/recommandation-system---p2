# Recommendation System Implementation Plan

## Overview
I've created a comprehensive movie recommendation system with two different approaches:
1. **Content-Based Filtering** (`app.py`) - Recommends movies based on genre similarity
2. **Collaborative Filtering** (`collaborative_filtering.py`) - Recommends movies based on user similarity

## Files Created

### 1. `app.py` - Content-Based Recommendation System
**Features:**
- Uses TF-IDF vectorization to convert movie genres into numerical features
- Calculates cosine similarity between user preferences and movie genres
- Provides top 5 recommendations based on genre matching
- Interactive command-line interface

**How it works:**
1. Creates a dataset of 50 popular movies with genre information
2. Vectorizes genres using TF-IDF
3. Takes user input for preferred genres
4. Calculates similarity scores
5. Returns top matching movies

### 2. `collaborative_filtering.py` - Collaborative Filtering System
**Features:**
- Uses user-item rating matrix
- Calculates user similarity using cosine similarity
- Predicts ratings for unrated movies based on similar users
- Interactive rating collection from user

**How it works:**
1. Creates sample user-movie rating data
2. Calculates similarity between users
3. Collects ratings from current user
4. Finds similar users and predicts ratings
5. Returns top recommendations

### 3. `requirements.txt`
Lists required Python packages:
- pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- numpy: Numerical computing

### 4. `README.md`
Project documentation and usage instructions

## Technical Implementation Details

### Content-Based Filtering Approach
- **TF-IDF Vectorization**: Converts text genres into numerical vectors
- **Cosine Similarity**: Measures similarity between user preferences and movie genres
- **Sample Data**: 50 movies with diverse genres

### Collaborative Filtering Approach
- **User-Item Matrix**: Matrix of users vs movie ratings
- **Similarity Calculation**: Cosine similarity between user rating vectors
- **Rating Prediction**: Weighted average of similar users' ratings

## Usage Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run content-based filtering:**
   ```bash
   python app.py
   ```

3. **Run collaborative filtering:**
   ```bash
   python collaborative_filtering.py
   ```

## Sample Data
- **Movies**: 50 popular films across various genres
- **Genres**: Action, Adventure, Comedy, Crime, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, Animation, Biography, History, Music, War, Family

## Future Enhancements
- Add more sophisticated algorithms (SVD, neural networks)
- Integrate with real movie databases (IMDb, TMDB)
- Create web interface
- Add user persistence and history
- Implement hybrid recommendation approaches

## Testing
Both systems include error handling and user-friendly interfaces for easy testing and demonstration.
