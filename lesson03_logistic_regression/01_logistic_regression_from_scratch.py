# Lesson 3: Logistic Regression from Scratch Using NumPy

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Generate synthetic binary classification data
# ------------------------------------------------------------

def generate_binary_classification_data(num_samples, noise_std, random_seed=0):
    """
    Generate a 2D binary classification dataset.

    We generate two Gaussian clusters:
        class 0 centered near (-2, -2)
        class 1 centered near ( 2,  2)

    Parameters
    ----------
    num_samples : int
        Total number of data points.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (num_samples, 2)
        Feature matrix.
    y : np.ndarray, shape (num_samples,)
        Binary labels in {0, 1}.
    """

    rng = np.random.default_rng(random_seed)

    num_class0 = num_samples // 2
    num_class1 = num_samples - num_class0

    X0 = rng.normal(loc=-2.0, scale=noise_std, size=(num_class0, 2))
    X1 = rng.normal(loc=2.0, scale=noise_std, size=(num_class1, 2))

    y0 = np.zeros(num_class0)
    y1 = np.ones(num_class1)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # Shuffle the dataset
    permutation = rng.permutation(num_samples)
    X = X[permutation]
    y = y[permutation]

    return X, y


# ------------------------------------------------------------
# 2. Define sigmoid function
# ------------------------------------------------------------

def sigmoid(z):
    """
    Compute sigmoid(z) = 1 / (1 + exp(-z)).

    For this small example, the direct implementation is fine.
    Later, for large models, numerical stability becomes more important.
    """

    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------
# 3. Define prediction functions
# ------------------------------------------------------------

def predict_probabilities(X, w, b):
    """
    Compute predicted probabilities:

        p = sigmoid(Xw + b)

    X shape: (n, d)
    w shape: (d,)
    b shape: scalar

    output shape: (n,)
    """

    logits = X @ w + b
    probabilities = sigmoid(logits)

    return probabilities


def predict_labels(X, w, b, threshold=0.5):
    """
    Convert predicted probabilities into binary labels.

    If p >= threshold, predict class 1.
    Otherwise, predict class 0.
    """

    probabilities = predict_probabilities(X, w, b)
    labels = (probabilities >= threshold).astype(int)

    return labels


# ------------------------------------------------------------
# 4. Define binary cross-entropy loss
# ------------------------------------------------------------

def binary_cross_entropy(probabilities, y_true):
    """
    Binary cross-entropy loss:

        L = -mean(y log(p) + (1-y) log(1-p))

    We clip probabilities to avoid log(0).
    """

    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0 - eps)

    loss = -np.mean(
        y_true * np.log(probabilities)
        + (1.0 - y_true) * np.log(1.0 - probabilities)
    )

    return loss


# ------------------------------------------------------------
# 5. Compute gradients
# ------------------------------------------------------------

def compute_gradients(X, y, w, b):
    """
    Compute gradients for logistic regression.

    For binary cross-entropy loss:

        errors = probabilities - y
        grad_w = (1/n) X^T errors
        grad_b = (1/n) sum(errors)
    """

    num_samples = X.shape[0]

    probabilities = predict_probabilities(X, w, b)
    errors = probabilities - y

    grad_w = (1.0 / num_samples) * (X.T @ errors)
    grad_b = (1.0 / num_samples) * np.sum(errors)

    return grad_w, grad_b


# ------------------------------------------------------------
# 6. Accuracy
# ------------------------------------------------------------

def accuracy_score(y_pred, y_true):
    """
    Compute classification accuracy.
    """

    return np.mean(y_pred == y_true)


# ------------------------------------------------------------
# 7. Train logistic regression using gradient descent
# ------------------------------------------------------------

def train_logistic_regression(X, y, learning_rate, num_iterations):
    """
    Train logistic regression using full-batch gradient descent.
    """

    num_features = X.shape[1]

    w = np.zeros(num_features)
    b = 0.0

    loss_history = []
    accuracy_history = []

    for iteration in range(num_iterations):
        # Forward pass
        probabilities = predict_probabilities(X, w, b)
        loss = binary_cross_entropy(probabilities, y)

        predicted_labels = predict_labels(X, w, b)
        accuracy = accuracy_score(predicted_labels, y)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        # Backward pass
        grad_w, grad_b = compute_gradients(X, y, w, b)

        # Gradient descent update
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        if iteration % 100 == 0:
            print(
                f"Iteration {iteration:4d} | "
                f"Loss = {loss:.6f} | "
                f"Accuracy = {accuracy:.4f}"
            )

    return w, b, loss_history, accuracy_history


# ------------------------------------------------------------
# 8. Plot data and decision boundary
# ------------------------------------------------------------

def plot_decision_boundary(X, y, w, b):
    """
    Plot the 2D data and the learned decision boundary.

    The decision boundary is where:

        x^T w + b = 0

    For 2D data:

        w0 * x1 + w1 * x2 + b = 0

    Solving for x2:

        x2 = -(w0 * x1 + b) / w1
    """

    plt.figure()

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

    x_values = np.linspace(X[:, 0].min() - 1.0, X[:, 0].max() + 1.0, 200)

    if abs(w[1]) > 1e-12:
        y_values = -(w[0] * x_values + b) / w[1]
        plt.plot(x_values, y_values, label="Decision boundary")
    else:
        # Vertical decision boundary
        x_boundary = -b / w[0]
        plt.axvline(x_boundary, label="Decision boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic regression decision boundary")
    plt.legend()
    plt.show()


def plot_training_curves(loss_history, accuracy_history):
    """
    Plot loss and accuracy curves separately.
    """

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Training loss")
    plt.show()

    plt.figure()
    plt.plot(accuracy_history)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.show()


# ------------------------------------------------------------
# 9. Main script
# ------------------------------------------------------------

def main():
    # Dataset settings
    num_samples = 200
    noise_std = 1.0

    X, y = generate_binary_classification_data(
        num_samples=num_samples,
        noise_std=noise_std,
        random_seed=0,
    )

    print("Data shapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nFirst five examples:")
    print("X[:5] =")
    print(X[:5])
    print("y[:5] =")
    print(y[:5])

    # Training hyperparameters
    learning_rate = 0.01
    num_iterations = 1000

    print("\nTraining logistic regression from scratch...\n")

    w_learned, b_learned, loss_history, accuracy_history = train_logistic_regression(
        X=X,
        y=y,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
    )

    print("\nLearned parameters:")
    print("w_learned:", w_learned)
    print("b_learned:", b_learned)

    final_probabilities = predict_probabilities(X, w_learned, b_learned)
    final_labels = predict_labels(X, w_learned, b_learned)

    final_loss = binary_cross_entropy(final_probabilities, y)
    final_accuracy = accuracy_score(final_labels, y)

    print("\nFinal performance:")
    print("Final loss:", final_loss)
    print("Final accuracy:", final_accuracy)

    print("\nFirst ten predicted probabilities:")
    print(final_probabilities[:10])

    print("\nFirst ten predicted labels:")
    print(final_labels[:10])

    print("\nFirst ten true labels:")
    print(y[:10].astype(int))

    plot_training_curves(loss_history, accuracy_history)
    plot_decision_boundary(X, y, w_learned, b_learned)


if __name__ == "__main__":
    main()