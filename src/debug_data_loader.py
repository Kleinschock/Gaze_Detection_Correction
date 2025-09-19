import os
import sys

# Ensure the parent directory is in the Python path to allow for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import GestureDataset
from src.config import DATA_ROOT

def main():
    """
    Initializes the GestureDataset in debug mode to analyze the data loading process.
    """
    print("--- Running Data Loader in Debug Mode ---")
    # We pass debug=True to activate the detailed logging.
    GestureDataset(data_root=DATA_ROOT, debug=True)
    print("\n--- Debug Run Finished ---")

if __name__ == '__main__':
    main()
