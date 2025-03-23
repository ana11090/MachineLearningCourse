import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = np.loadtxt('Assignment2data.txt', delimiter=',')
X = data[:, :-1]  # Features: size and number of bedrooms
y = data[:, -1]   # Target: house price
m = y.shape[0]    # Number of training examples

# ----------------------
# Step 1: Feature Normalization
# ----------------------
def normalize_features(X):
    """
    Normalize the features using mean and standard deviation.
    Formula for normalization:
        x_norm = (x - μ) / σ
    where:
        μ = mean of feature values
        σ = standard deviation of feature values
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X, mu, sigma = normalize_features(X)

# Add intercept term to X
X = np.hstack([np.ones((m, 1)), X])  # Add a column of ones to X

# Plot the normalized data
plt.figure()
plt.scatter(X[:, 1], y, color='blue', label="Size vs. Price")
plt.scatter(X[:, 2], y, color='green', label="Bedrooms vs. Price")
plt.xlabel("Normalized Features")
plt.ylabel("Price")
plt.title("Normalized Features vs. Price")
plt.legend()
plt.show()

# ----------------------
# Step 2: Vectorized Cost Function
# ----------------------
def compute_cost_vectorized(X, y, w0, w1, w2):
    """
    Compute cost for linear regression.
    Formula for cost function:
        J(w) = (1 / (2 * m)) * Σ [h_w(x^(i)) - y^(i)]^2
    where:
        h_w(x^(i)) = w0 + w1 * x1 + w2 * x2
    """
    m = y.shape[0]
    predictions = w0 + w1 * X[:, 1] + w2 * X[:, 2]
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost

# ----------------------
# Step 3: Vectorized Gradient Descent
# ----------------------
def gradient_descent_vectorized(X, y, w0, w1, w2, alpha, num_iters):
    """
    Perform gradient descent to learn w.
    Gradient descent update rules for w0, w1, w2:
        w0 := w0 - α * (1 / m) * Σ [(h_w(x^(i)) - y^(i))]
        w1 := w1 - α * (1 / m) * Σ [(h_w(x^(i)) - y^(i)) * x1^(i)]
        w2 := w2 - α * (1 / m) * Σ [(h_w(x^(i)) - y^(i)) * x2^(i)]
    """
    m = y.shape[0]
    J_history = []  # To store cost at each iteration
    
    for i in range(num_iters):
        predictions = w0 + w1 * X[:, 1] + w2 * X[:, 2]
        errors = predictions - y
        
        # Compute gradients
        grad_w0 = (1 / m) * np.sum(errors)
        grad_w1 = (1 / m) * np.dot(errors, X[:, 1])
        grad_w2 = (1 / m) * np.dot(errors, X[:, 2])
        
        # Update weights
        w0 -= alpha * grad_w0
        w1 -= alpha * grad_w1
        w2 -= alpha * grad_w2
        
        # Save the cost J in every iteration
        cost = compute_cost_vectorized(X, y, w0, w1, w2)
        J_history.append(cost)
        
        # Print intermediate results
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {cost:.4f},  w1 = {w1:.4f}, w2 = {w2:.4f}")
    
    return w0, w1, w2, J_history

# Initialize parameters
w0, w1, w2 = 0, 0, 0  # Initialize weights (including bias term)
alpha = 0.01          # Learning rate
iterations = 400      # Number of iterations

# Run gradient descent
print("Running gradient descent...")
w0_final, w1_final, w2_final, J_history = gradient_descent_vectorized(X, y, w0, w1, w2, alpha, iterations)

# Print final values
print(f"Final values: w1 = {w1_final:.10f}, w2 = {w2_final:.10f}")

# ----------------------
# Step 4: Visualizations
# ----------------------

# 1. Cost Function over Iterations
plt.figure()
plt.plot(range(iterations), J_history, color='purple')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function over Iterations")
plt.show()

# 2. Predictions vs. Actual
predictions = w0_final + w1_final * X[:, 1] + w2_final * X[:, 2]  # Compute predictions
plt.figure()
plt.scatter(y, predictions, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label="Perfect Fit")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predictions vs. Actual")
plt.legend()
plt.show()

# 3. Residuals Plot (Predicted - Actual)
residuals = predictions - y
plt.figure()
plt.scatter(predictions, residuals, color='orange')
plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Predicted - Actual)")
plt.title("Residuals Plot")
plt.legend()
plt.show()

# 4. 3D Visualization of Features and Target
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], y, color='blue', label='Actual Prices')
ax.set_xlabel('Normalized Size (x1)')
ax.set_ylabel('Normalized Bedrooms (x2)')
ax.set_zlabel('Price')
plt.title("3D Visualization of Features and Target")
plt.legend()
plt.show()

# ----------------------
# Step 5: Prediction for Specific Input
# ----------------------
house_features = np.array([1650, 3])  # Input features
house_features_norm = (house_features - mu) / sigma  # Normalize input
predicted_price = w0_final + w1_final * house_features_norm[0] + w2_final * house_features_norm[1]
print(f"Predicted price of a 1650 sqft house with 3 bedrooms: ${predicted_price:.2f}")
