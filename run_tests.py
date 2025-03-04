"""
Main script to run all unit tests
"""
import unittest
import sys
import os

def run_tests():
    """Run all test modules in the tests directory"""
    # Make sure tests directory is in path
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    
    # Make sure project root is in path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)