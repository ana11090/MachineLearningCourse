import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Step 1: Load Dataset
# ----------------------
# Load dataset from file (modify file path as necessary)
data = np.loadtxt('Assignment3data1.txt', delimiter=',')
X = data[:, :-1]  # Features: Exam scores
y = data[:, -1]   # Target: Admission decision (0 or 1)
m = y.shape[0]    # Number of training examples

# ----------------------
# Step 1.1: Visualize x1, xn, y to find a pattern
# ----------------------
plt.figure()
admitted = y == 1
not_admitted = y == 0

plt.scatter(X[admitted, 0], X[admitted, 1], color='blue', label="Admitted")
plt.scatter(X[not_admitted, 0], X[not_admitted, 1], color='red', label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.title("Visualization of Exam Scores (x1, xn) and Admission Decision (y)")
plt.legend() 
plt.show()

# ----------------------
# Step 2: Feature Normalization
# ----------------------
def normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X, mu, sigma = normalize_features(X)

# Add intercept term to X
X = np.hstack([np.ones((m, 1)), X])  # Add a column of ones for bias term

# ----------------------
# Step 3: Sigmoid Function
# ----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ----------------------
# Step 4: Logistic Regression Cost Function
# ----------------------
def compute_cost(X, y, w0, w1, w2):
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
# Step 5: Gradient Descent
# ----------------------
def gradient_descent(X, y, w0, w1, w2, alpha, num_iters):
    """
    Perform gradient descent to learn w0, w1, w2.
    """
    m = y.shape[0]
    J_history = []

    for i in range(num_iters):
        # Predictions
        predictions = sigmoid(w0 * X[:, 0] + w1 * X[:, 1] + w2 * X[:, 2])
        errors = predictions - y

        # Gradients
        grad_w0 = (1 / m) * np.sum(errors)
        grad_w1 = (1 / m) * np.dot(errors, X[:, 1])
        grad_w2 = (1 / m) * np.dot(errors, X[:, 2])

        # Update weights
        w0 -= alpha * grad_w0
        w1 -= alpha * grad_w1
        w2 -= alpha * grad_w2

        # Compute cost
        cost = compute_cost(X, y, w0, w1, w2)
        J_history.append(cost)

        # Print intermediate results
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost = {cost:.4f},   w1 = {w1:.4f}, w2 = {w2:.4f}")

    return w0, w1, w2, J_history

# ----------------------
# Step 6: Test Convergence with Varying Learning Rates
# ----------------------
learning_rates = [0.01, 0.001, 0.0001, 0.00001]
results = {}

for alpha in learning_rates:
    print(f"Testing learning rate: {alpha}")
    w0, w1, w2 = 0, 0, 0  # Reinitialize weights for each test
    w0, w1, w2, J_history = gradient_descent(X, y, w0, w1, w2, alpha, 400)
    results[alpha] = {
        "w0": w0, "w1": w1, "w2": w2, "cost": J_history[-1], "history": J_history
    }

    # Plot convergence for each learning rate
    plt.plot(range(len(J_history)), J_history, label=f"alpha={alpha}")

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Convergence for Different Learning Rates")
plt.legend()
plt.show()

# Display results for the best learning rate
best_alpha = min(results, key=lambda x: results[x]["cost"])
best_result = results[best_alpha]
print(f"Best learning rate: {best_alpha}")
print(f"Final values:  w1 = {best_result['w1']:.4f}, w2 = {best_result['w2']:.4f}")

# ----------------------
# Step 7: Visualize Decision Boundary
# ----------------------
x_values = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
y_values = -(best_result['w0'] + best_result['w1'] * x_values) / best_result['w2']

plt.figure()
plt.scatter(X[admitted, 1], X[admitted, 2], color='blue', label="Admitted")
plt.scatter(X[not_admitted, 1], X[not_admitted, 2], color='red', label="Not Admitted")
plt.plot(x_values, y_values, color='green', label="Decision Boundary")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()

# ----------------------
# Step 8: Predict Admission Probability
# ----------------------
def predict_probability(w0, w1, w2, features, mu, sigma):
    """
    Predict the probability of admission for given features.
    """
    features_norm = (features - mu) / sigma  # Normalize input
    features_with_bias = np.hstack([1, features_norm])  # Add intercept term
    return sigmoid(w0 * features_with_bias[0] + w1 * features_with_bias[1] + w2 * features_with_bias[2])

student_scores = np.array([50, 100])  # Example: Exam scores
probability = predict_probability(best_result['w0'], best_result['w1'], best_result['w2'], student_scores, mu, sigma)
result = "Accepted" if probability >= 0.5 else "Rejected"
print(f"Probability of admission for scores 50 and 100: {probability:.4f}. Admission decision: {result}")
