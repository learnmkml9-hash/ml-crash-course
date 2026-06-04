# Lesson 5: A Two-Layer Neural Network from Scratch Using NumPy

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Generate nonlinear multiclass data
# ------------------------------------------------------------

def generate_spiral_data(num_samples_per_class, num_classes, noise_std, random_seed=0):
    """
    Generate a 2D nonlinear multiclass spiral dataset.

    This dataset is intentionally nonlinear. A linear softmax classifier
    will struggle, but a neural network with a hidden layer can learn
    a nonlinear decision boundary.

    Parameters
    ----------
    num_samples_per_class : int
        Number of samples per class.
    num_classes : int
        Number of classes.
    noise_std : float
        Amount of angular noise.
    random_seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n, 2)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Integer class labels.
    """

    rng = np.random.default_rng(random_seed)

    num_samples = num_samples_per_class * num_classes
    X = np.zeros((num_samples, 2))
    y = np.zeros(num_samples, dtype=int)

    for class_index in range(num_classes):
        start = class_index * num_samples_per_class
        end = start + num_samples_per_class

        r = np.linspace(0.0, 1.0, num_samples_per_class)
        theta = (
            np.linspace(class_index * 4.0, (class_index + 1) * 4.0, num_samples_per_class)
            + noise_std * rng.normal(size=num_samples_per_class)
        )

        X[start:end, 0] = r * np.sin(theta)
        X[start:end, 1] = r * np.cos(theta)
        y[start:end] = class_index

    permutation = rng.permutation(num_samples)
    X = X[permutation]
    y = y[permutation]

    return X, y


# ------------------------------------------------------------
# 2. One-hot encoding
# ------------------------------------------------------------

def one_hot_encode(y, num_classes):
    """
    Convert integer labels into one-hot vectors.

    y shape: (n,)
    output shape: (n, K)
    """

    num_samples = y.shape[0]
    Y = np.zeros((num_samples, num_classes))
    Y[np.arange(num_samples), y] = 1.0

    return Y


# ------------------------------------------------------------
# 3. Activation functions
# ------------------------------------------------------------

def relu(z):
    """
    ReLU activation:

        ReLU(z) = max(0, z)
    """

    return np.maximum(0.0, z)


def relu_derivative(z):
    """
    Derivative of ReLU.

    The derivative is:
        1 if z > 0
        0 if z <= 0

    At z = 0, the derivative is technically undefined.
    In code, we set it to 0.
    """

    return (z > 0.0).astype(float)


# ------------------------------------------------------------
# 4. Softmax
# ------------------------------------------------------------

def softmax(logits):
    """
    Numerically stable row-wise softmax.

    logits shape: (n, K)
    output shape: (n, K)
    """

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probabilities


# ------------------------------------------------------------
# 5. Initialize parameters
# ------------------------------------------------------------

def initialize_parameters(input_dim, hidden_dim, num_classes, random_seed=0):
    """
    Initialize neural network parameters.

    W1 shape: (input_dim, hidden_dim)
    b1 shape: (hidden_dim,)
    W2 shape: (hidden_dim, num_classes)
    b2 shape: (num_classes,)
    """

    rng = np.random.default_rng(random_seed)

    params = {
        "W1": 0.1 * rng.normal(size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": 0.1 * rng.normal(size=(hidden_dim, num_classes)),
        "b2": np.zeros(num_classes),
    }

    return params


# ------------------------------------------------------------
# 6. Forward pass
# ------------------------------------------------------------

def forward_pass(X, params):
    """
    Compute the forward pass of the two-layer neural network.

    X -> linear layer -> ReLU -> linear layer -> softmax

    Returns
    -------
    probabilities : np.ndarray, shape (n, K)
        Class probabilities.
    cache : dict
        Intermediate quantities needed for backpropagation.
    """

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    assert X.ndim == 2
    assert W1.ndim == 2
    assert W2.ndim == 2
    assert b1.ndim == 1
    assert b2.ndim == 1
    assert X.shape[1] == W1.shape[0]
    assert W1.shape[1] == b1.shape[0]
    assert W2.shape[0] == W1.shape[1]
    assert W2.shape[1] == b2.shape[0]

    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    probabilities = softmax(Z2)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "probabilities": probabilities,
    }

    return probabilities, cache


# ------------------------------------------------------------
# 7. Loss and accuracy
# ------------------------------------------------------------

def cross_entropy_loss(probabilities, Y):
    """
    Multiclass cross-entropy loss.

    probabilities shape: (n, K)
    Y shape: (n, K)
    """

    assert probabilities.shape == Y.shape

    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0 - eps)

    loss = -np.mean(np.sum(Y * np.log(probabilities), axis=1))

    return loss


def l2_regularization_loss(params, regularization_strength):
    """
    Compute the L2 regularization contribution to the loss.

    Only the weight matrices W1 and W2 are regularized; biases are not.
    """

    W1 = params["W1"]
    W2 = params["W2"]

    return 0.5 * regularization_strength * (np.sum(W1 ** 2) + np.sum(W2 ** 2))


def compute_loss(probabilities, Y, params, regularization_strength=0.0):
    """
    Compute the total loss including cross-entropy and L2 regularization.
    """

    loss = cross_entropy_loss(probabilities, Y)
    loss += l2_regularization_loss(params, regularization_strength)

    return loss


def predict_labels(X, params):
    """
    Predict class labels.
    """

    probabilities, _ = forward_pass(X, params)
    labels = np.argmax(probabilities, axis=1)

    return labels


def accuracy_score(y_pred, y_true):
    """
    Classification accuracy.
    """

    return np.mean(y_pred == y_true)


# ------------------------------------------------------------
# 8. Backward pass
# ------------------------------------------------------------

def backward_pass(params, cache, Y, regularization_strength=0.0):
    """
    Backpropagation for the two-layer neural network.

    Forward equations:
        Z1 = X W1 + b1
        A1 = ReLU(Z1)
        Z2 = A1 W2 + b2
        P  = softmax(Z2)

    Loss:
        cross-entropy(P, Y) + 0.5 * reg_strength * (||W1||^2 + ||W2||^2)

    Important gradient:
        dZ2 = (P - Y) / n
    """

    W1 = params["W1"]
    W2 = params["W2"]

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    probabilities = cache["probabilities"]

    num_samples = X.shape[0]

    assert probabilities.shape == Y.shape

    # Gradient through softmax + cross-entropy
    dZ2 = (probabilities - Y) / num_samples

    # Gradients for second layer
    grad_W2 = A1.T @ dZ2
    grad_b2 = np.sum(dZ2, axis=0)

    # Add L2 regularization gradient for W2
    grad_W2 += regularization_strength * W2

    # Backpropagate into hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    # Gradients for first layer
    grad_W1 = X.T @ dZ1
    grad_b1 = np.sum(dZ1, axis=0)

    # Add L2 regularization gradient for W1
    grad_W1 += regularization_strength * W1

    grads = {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }

    return grads


# ------------------------------------------------------------
# 9. Parameter update
# ------------------------------------------------------------

def update_parameters(params, grads, learning_rate):
    """
    Gradient descent update.
    """

    for key in params:
        params[key] = params[key] - learning_rate * grads[key]

    return params


# ------------------------------------------------------------
# 10. Training loop
# ------------------------------------------------------------

def train_two_layer_nn(
    X,
    y,
    num_classes,
    hidden_dim,
    learning_rate,
    num_iterations,
    regularization_strength=0.0,
    random_seed=0,
):
    """
    Train a two-layer neural network using full-batch gradient descent.
    """

    input_dim = X.shape[1]
    Y = one_hot_encode(y, num_classes)

    params = initialize_parameters(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        random_seed=random_seed,
    )

    loss_history = []
    accuracy_history = []

    for iteration in range(num_iterations):
        probabilities, cache = forward_pass(X, params)
        loss = compute_loss(probabilities, Y, params, regularization_strength)

        y_pred = np.argmax(probabilities, axis=1)
        accuracy = accuracy_score(y_pred, y)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        grads = backward_pass(params, cache, Y, regularization_strength)
        params = update_parameters(params, grads, learning_rate)

        if iteration % 500 == 0:
            print(
                f"Iteration {iteration:5d} | "
                f"Loss = {loss:.6f} | "
                f"Accuracy = {accuracy:.4f}"
            )

    return params, loss_history, accuracy_history


# ------------------------------------------------------------
# 11. Plotting
# ------------------------------------------------------------

def plot_training_curves(loss_history, accuracy_history):
    """
    Plot loss and accuracy curves.
    """

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training loss")
    plt.show()

    plt.figure()
    plt.plot(accuracy_history)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.show()


def plot_decision_regions(X, y, params):
    """
    Plot learned decision regions in 2D.
    """

    x_min = X[:, 0].min() - 0.2
    x_max = X[:, 0].max() + 0.2
    y_min = X[:, 1].min() - 0.2
    y_max = X[:, 1].max() + 0.2

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_predictions = predict_labels(grid_points, params)
    grid_predictions = grid_predictions.reshape(grid_x.shape)

    plt.figure()
    plt.contourf(grid_x, grid_y, grid_predictions, alpha=0.3)

    classes = np.unique(y)

    for class_index in classes:
        plt.scatter(
            X[y == class_index, 0],
            X[y == class_index, 1],
            label=f"Class {class_index}",
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Two-layer neural network decision regions")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 12. Main script
# ------------------------------------------------------------

def main():
    # Dataset settings
    num_samples_per_class = 100
    num_classes = 3
    noise_std = 0.25

    X, y = generate_spiral_data(
        num_samples_per_class=num_samples_per_class,
        num_classes=num_classes,
        noise_std=noise_std,
        random_seed=0,
    )

    print("Data shapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nLabel values:")
    print(np.unique(y))

    print("\nFirst five examples:")
    print("X[:5] =")
    print(X[:5])
    print("y[:5] =")
    print(y[:5])

    # Model and training hyperparameters
    hidden_dim = 128
    learning_rate = 1.0
    num_iterations = 5000
    regularization_strength = 1e-3

    print("\nTraining two-layer neural network from scratch...\n")

    params, loss_history, accuracy_history = train_two_layer_nn(
        X=X,
        y=y,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        regularization_strength=regularization_strength,
        random_seed=0,
    )

    final_probabilities, _ = forward_pass(X, params)
    final_predictions = np.argmax(final_probabilities, axis=1)

    final_loss = compute_loss(
        final_probabilities,
        one_hot_encode(y, num_classes),
        params,
        regularization_strength=regularization_strength,
    )
    final_accuracy = accuracy_score(final_predictions, y)

    print("\nFinal performance:")
    print("Final loss:", final_loss)
    print("Final accuracy:", final_accuracy)

    print("\nParameter shapes:")
    print("W1:", params["W1"].shape)
    print("b1:", params["b1"].shape)
    print("W2:", params["W2"].shape)
    print("b2:", params["b2"].shape)

    print("\nFirst five predicted probability vectors:")
    print(final_probabilities[:5])

    print("\nProbability row sums for first five examples:")
    print(np.sum(final_probabilities[:5], axis=1))

    print("\nFirst five predicted labels:")
    print(final_predictions[:5])

    print("\nFirst five true labels:")
    print(y[:5])

    plot_training_curves(loss_history, accuracy_history)
    plot_decision_regions(X, y, params)


if __name__ == "__main__":
    main()