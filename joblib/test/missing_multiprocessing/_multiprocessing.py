"""
Simulate a missing _multiprocessing module by raising an ImportError.
test_missing_multiprocessing adds this folder to the path.
"""
raise ImportError("No _multiprocessing module!")
