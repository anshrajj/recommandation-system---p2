#!/usr/bin/env python3
"""
Simple test script to verify sklearn installation
"""

def test_sklearn():
    """Test if sklearn is properly installed"""
    print("Testing scikit-learn installation...")
    
    try:
        import sklearn
        print("scikit-learn version:", sklearn.__version__)
        
        # Test specific modules used in the project
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("TfidfVectorizer imported successfully")
        
        from sklearn.metrics.pairwise import cosine_similarity
        print("cosine_similarity imported successfully")
        
        # Test basic functionality
        print("Testing basic functionality...")
        vectorizer = TfidfVectorizer()
        texts = ["action sci-fi", "drama romance", "comedy"]
        X = vectorizer.fit_transform(texts)
        print("TF-IDF matrix shape:", X.shape)
        
        similarity = cosine_similarity(X[0:1], X[1:2])
        print("Cosine similarity computed:", similarity[0][0])
        
        print("All sklearn modules are working correctly!")
        return True
        
    except ImportError as e:
        print("Import error:", e)
        return False
    except Exception as e:
        print("Error during testing:", e)
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SKLEARN INSTALLATION TEST")
    print("=" * 50)
    
    success = test_sklearn()
    
    print("=" * 50)
    if success:
        print("SUCCESS: Installation successful! All required sklearn modules are working.")
    else:
        print("FAILED: Installation failed. Please check your Python environment.")
    print("=" * 50)
