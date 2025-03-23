import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------
# Step 1: Load and Visualize Dataset
# ----------------------
data = np.loadtxt('Assignment3data2.txt', delimiter=',')
X = data[:, :-1]  # Features: Test results for the microchips
y = data[:, -1]   # Target: Passed (1) or Rejected (0)
m = y.shape[0]    # Number of training examples

# Plot the data
plt.figure()
passed = y == 1
rejected = y == 0

plt.scatter(X[passed, 0], X[passed, 1], color='blue', label="Passed")
plt.scatter(X[rejected, 0], X[rejected, 1], color='red', label="Rejected")
plt.xlabel("Test 1 Result")
plt.ylabel("Test 2 Result")
plt.title("Microchip Test Results")
plt.legend()
plt.grid()
plt.show()

# ----------------------
# Step 2: Feature Mapping for Polynomial Terms
# ----------------------
def map_feature(X1, X2, degree=6):
    """
    Maps two input features to polynomial features up to the given degree.
    """
    out = [np.ones(X1.shape[0])]  # Add a column of ones for the bias term (intercept)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
    return np.stack(out, axis=1)

# Map features to higher dimensions (degree 6)
X_mapped = map_feature(X[:, 0], X[:, 1], degree=6)
print(f"Feature mapping completed. Shape of X_mapped: {X_mapped.shape}")

# ----------------------
# Step 3: Sigmoid Function
# ----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ----------------------
# Step 4: Cost Function with Regularization
# ----------------------
def compute_cost_with_regularization(theta, X, y, reg_param):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    reg_term = (reg_param / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg_term

# Gradient computation with regularization
def compute_gradient_with_regularization(theta, X, y, reg_param):
    m = len(y)
    h = sigmoid(X @ theta)
    gradient = (1 / m) * (X.T @ (h - y))
    gradient[1:] += (reg_param / m) * theta[1:]
    return gradient

# ----------------------
# Step 5: Optimize Parameters
# ----------------------
initial_theta = np.zeros(X_mapped.shape[1])
reg_param = 1

res = minimize(
    fun=compute_cost_with_regularization,
    x0=initial_theta,
    args=(X_mapped, y, reg_param),
    jac=compute_gradient_with_regularization,
    options={'maxiter': 400}
)
optimal_theta = res.x
final_cost = compute_cost_with_regularization(optimal_theta, X_mapped, y, reg_param)

print("\n=== Final Training Results ===")
print(f"Final Cost: {final_cost:.4f}")
print(f"Optimal Weights (theta):")
for i, theta in enumerate(optimal_theta):
    print(f"  Theta[{i}]: {theta:.4f}")

# ----------------------
# Step 6: Visualize the Decision Boundary
# ----------------------
def plot_decision_boundary(theta, X, y, degree=6):
    plt.figure()

    # Plot data points
    passed = y == 1
    rejected = y == 0
    plt.scatter(X[passed, 0], X[passed, 1], color='blue', label="Passed")
    plt.scatter(X[rejected, 0], X[rejected, 1], color='red', label="Rejected")

    # Generate grid for boundary
    u = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    v = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = map_feature(np.array([u[i]]), np.array([v[j]]), degree).dot(theta)

    z = z.T
    plt.contour(u, v, z, levels=[0], colors='green', linewidths=2)

    plt.xlabel("Test 1 Result")
    plt.ylabel("Test 2 Result")
    plt.title("Decision Boundary")
    plt.legend()
    plt.grid()
    plt.show()

plot_decision_boundary(optimal_theta, X, y, degree=6)

# ----------------------
# Step 7: Accuracy Calculation
# ----------------------
def predict(theta, X):
    return sigmoid(X @ theta) >= 0.5

accuracy = np.mean(predict(optimal_theta, X_mapped) == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")

# ----------------------
# Step 8: Explore Regularization Parameter
# ----------------------
print("\n=== Exploring Regularization Parameters ===")
learning_rates = [0.001, 0.01, 0.1, 1]
final_costs = []
final_weights = []
accuracies = []  # To store accuracy for each learning rate

for lr in learning_rates:
    res = minimize(
        fun=compute_cost_with_regularization,
        x0=initial_theta,
        args=(X_mapped, y, lr),
        jac=compute_gradient_with_regularization,
        options={'maxiter': 400}
    )
    cost = compute_cost_with_regularization(res.x, X_mapped, y, lr)
    predictions = predict(res.x, X_mapped)
    accuracy = np.mean(predictions == y) * 100
    final_costs.append(cost)
    final_weights.append(res.x)
    accuracies.append(accuracy)
    print(f"Learning Rate: {lr:.3f}, Final Cost: {cost:.4f}, Accuracy: {accuracy:.2f}%, Weights: {res.x[:5]}...")

best_lr_index = np.argmin(final_costs)
best_lr = learning_rates[best_lr_index]
best_weights = final_weights[best_lr_index]

print(f"\nBest Learning Rate: {best_lr}")
print(f"Final Cost with Best Learning Rate: {final_costs[best_lr_index]:.4f}")
print(f"Accuracy with Best Learning Rate: {accuracies[best_lr_index]:.2f}%")
print(f"Best Weights (First 5): {best_weights[:5]} ")


# ----------------------
# Step 9: Real Prediction Example
# ----------------------
test_scores = np.array([50, 100])
test_scores_mapped = map_feature(test_scores[0:1], test_scores[1:2], degree=6)
probability = sigmoid(test_scores_mapped @ optimal_theta)
decision = "Accepted" if probability >= 0.5 else "Rejected"

print("\n=== Real Prediction Example ===")
print(f"Test Scores: Test 1 = {test_scores[0]}, Test 2 = {test_scores[1]}")
print(f"Probability of Acceptance: {probability[0]:.4f}")
print(f"Admission Decision: {decision}")
