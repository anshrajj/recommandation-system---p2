#!/usr/bin/env python3
"""
Test script to verify sklearn installation and available modules
"""

def test_sklearn_installation():
    """Test if sklearn is properly installed and working"""
    print("Testing scikit-learn installation...")
    
    try:
        import sklearn
        print(f"✓ scikit-learn version: {sklearn.__version__}")
        
        # Test specific modules used in the project
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ TfidfVectorizer imported successfully")
        
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓ cosine_similarity imported successfully")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        vectorizer = TfidfVectorizer()
        texts = ["action sci-fi", "drama romance", "comedy"]
        X = vectorizer.fit_transform(texts)
        print(f"✓ TF-IDF matrix shape: {X.shape}")
        
        similarity = cosine_similarity(X[0:1], X[1:2])
        print(f"✓ Cosine similarity computed: {similarity[0][0]:.3f}")
        
        print("\n🎉 All sklearn modules are working correctly!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

def test_additional_sklearn_features():
    """Test additional sklearn features that could be useful"""
    print("\nTesting additional sklearn features...")
    
    try:
        # Text processing features
        from sklearn.feature_extraction.text import CountVectorizer
        print("✓ CountVectorizer available")
        
        # Similarity metrics
        from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
        print("✓ Additional distance metrics available")
        
        # Dimensionality reduction
        from sklearn.decomposition import PCA, TruncatedSVD
        print("✓ Dimensionality reduction methods available")
        
        # Clustering
        from sklearn.cluster import KMeans, DBSCAN
        print("✓ Clustering algorithms available")
        
        # Preprocessing
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        print("✓ Data preprocessing tools available")
        
        # Model selection
        from sklearn.model_selection import train_test_split, cross_val_score
        print("✓ Model selection tools available")
        
        print("✓ All additional sklearn features are available!")
        return True
        
    except ImportError as e:
        print(f"✗ Some features not available: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SKLEARN INSTALLATION TEST")
    print("=" * 50)
    
    success = test_sklearn_installation()
    if success:
        test_additional_sklearn_features()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Installation successful! All required sklearn modules are working.")
    else:
        print("❌ Installation failed. Please check your Python environment.")
    print("=" * 50)
