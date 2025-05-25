import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import NotFittedError

def main():
    try:
        # Step 1: Read N
        N = int(input("Enter the number of data points (N): "))
        if N <= 0:
            print("N must be a positive integer.")
            return

        # Step 2: Read k
        k = int(input("Enter the value of k for k-NN regression: "))
        if k <= 0:
            print("k must be a positive integer.")
            return
        if k > N:
            print("Error: k cannot be greater than N.")
            return

        # Step 3: Read N (x, y) points
        X_list = []
        y_list = []

        print("Enter the data points (x and y):")
        for i in range(N):
            x = float(input(f"Point {i+1} - x: "))
            y = float(input(f"Point {i+1} - y: "))
            X_list.append([x])  # Make x a list to match sklearn's expected 2D input
            y_list.append(y)

        # Convert to NumPy arrays
        X_np = np.array(X_list)
        y_np = np.array(y_list)

        # Step 4: Compute variance of y
        variance = np.var(y_np)
        print(f"\nVariance of labels in training data: {variance:.4f}")

        # Step 5: Read query input X
        query_x = float(input("\nEnter the input X to predict its Y using k-NN: "))
        query_point = np.array([[query_x]])

        # Step 6: Fit and predict using scikit-learn
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_np, y_np)
        prediction = model.predict(query_point)

        print(f"\nPredicted Y for X = {query_x} using k = {k}: {prediction[0]:.4f}")

    except ValueError:
        print("Invalid input. Please enter numeric values where required.")
    except NotFittedError:
        print("Model training failed.")

if __name__ == "__main__":
    main()
