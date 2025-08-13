# utils.py
def flatten_array(x):
    """Flatten a 2D array into 1D (needed for TfidfVectorizer in pipelines)."""
    return x.ravel()

