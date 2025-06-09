import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def get_positive_integer(prompt):
    """Helper function to get a positive integer from user input."""
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

def get_real_number(prompt):
    """Helper function to get a real number from user input."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def get_non_negative_integer(prompt):
    """Helper function to get a non-negative integer from user input."""
    while True:
        try:
            value = int(input(prompt))
            if value >= 0:
                return value
            else:
                print("Please enter a non-negative integer.")
        except ValueError:
            print("Please enter a valid integer.")

def collect_data_pairs(n, dataset_name):
    """
    Collect N (x, y) pairs from user input using NumPy arrays.
    
    Args:
        n (int): Number of pairs to collect
        dataset_name (str): Name of the dataset (for display purposes)
    
    Returns:
        tuple: (X_array, y_array) where X_array is features and y_array is labels
    """
    print(f"\n--- Collecting {dataset_name} Data ---")
    
    # Initialize NumPy arrays
    X_data = np.zeros(n, dtype=float)
    y_data = np.zeros(n, dtype=int)
    
    # Collect data pairs one by one
    for i in range(n):
        print(f"\nPair {i + 1}:")
        x_value = get_real_number(f"  Enter x value: ")
        y_value = get_non_negative_integer(f"  Enter y value (class label): ")
        
        # Store in NumPy arrays
        X_data[i] = x_value
        y_data[i] = y_value
    
    # Reshape X to be a 2D array (required by scikit-learn)
    X_data = X_data.reshape(-1, 1)
    
    return X_data, y_data

def find_best_k(X_train, y_train, X_test, y_test, k_range=(1, 11)):
    """
    Find the best k value for kNN classifier using hyperparameter search.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        X_test (np.array): Test features
        y_test (np.array): Test labels
        k_range (tuple): Range of k values to test (start, end)
    
    Returns:
        tuple: (best_k, best_accuracy, all_results)
    """
    print(f"\n--- Hyperparameter Search for k (range: {k_range[0]} to {k_range[1]-1}) ---")
    
    best_k = None
    best_accuracy = -1
    results = []
    
    # Try different k values
    for k in range(k_range[0], k_range[1]):
        # Check if k is not larger than training set size
        if k > len(X_train):
            print(f"k={k} is larger than training set size ({len(X_train)}), skipping...")
            continue
        
        # Create and train kNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = knn.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results.append((k, accuracy))
        
        print(f"k={k}: Test Accuracy = {accuracy:.4f}")
        
        # Update best k if this is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    return best_k, best_accuracy, results

def display_results(best_k, best_accuracy, all_results):
    """Display the final results in a nice format."""
    print(f"\n{'='*50}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*50}")
    
    print("\nAll k values tested:")
    for k, acc in all_results:
        marker = " ← BEST" if k == best_k else ""
        print(f"  k={k}: Accuracy = {acc:.4f}{marker}")
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS:")
    print(f"Best k value: {best_k}")
    print(f"Best test accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"{'='*50}")

def main():
    """Main function that orchestrates the entire program."""
    print("="*60)
    print("MINI kNN CLASSIFIER WITH HYPERPARAMETER SEARCH")
    print("="*60)
    
    # Step 1: Get training data
    print("\n🔹 STEP 1: Collect Training Data")
    N = get_positive_integer("Enter the number of training samples (N): ")
    X_train, y_train = collect_data_pairs(N, "Training Set")
    
    print(f"\nTraining data collected successfully!")
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Unique classes in training: {np.unique(y_train)}")
    
    # Step 2: Get test data
    print("\n🔹 STEP 2: Collect Test Data")
    M = get_positive_integer("Enter the number of test samples (M): ")
    X_test, y_test = collect_data_pairs(M, "Test Set")
    
    print(f"\nTest data collected successfully!")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Unique classes in test: {np.unique(y_test)}")
    
    # Step 3: Hyperparameter search
    print("\n🔹 STEP 3: Hyperparameter Search")
    
    # Check if we have enough training samples
    max_k = min(10, len(X_train))
    if max_k < 1:
        print("Error: Need at least 1 training sample!")
        return
    
    best_k, best_accuracy, all_results = find_best_k(
        X_train, y_train, X_test, y_test, k_range=(1, max_k + 1)
    )
    
    # Step 4: Display results
    print("\n🔹 STEP 4: Results")
    display_results(best_k, best_accuracy, all_results)
    
    # Optional: Show some additional analysis
    print(f"\nAdditional Information:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature range in training: [{np.min(X_train):.2f}, {np.max(X_train):.2f}]")
    print(f"Feature range in test: [{np.min(X_test):.2f}, {np.max(X_test):.2f}]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your input and try again.")
