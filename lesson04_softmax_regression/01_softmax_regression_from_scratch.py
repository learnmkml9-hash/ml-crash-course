# Lesson 4: Multiclass Classification with Softmax Regression

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Generate synthetic multiclass classification data
# ------------------------------------------------------------

def generate_multiclass_data(num_samples_per_class, noise_std, random_seed=0):
    """
    Generate a 2D, 3-class classification dataset.

    We create three Gaussian clusters in R^2.

    Parameters
    ----------
    num_samples_per_class : int
        Number of samples for each class.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n, 2)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Integer labels in {0, 1, 2}.
    """

    rng = np.random.default_rng(random_seed)

    centers = np.array([
        [-2.0, -2.0],
        [ 2.0, -2.0],
        [ 0.0,  2.0],
    ])

    X_list = []
    y_list = []

    num_classes = centers.shape[0]

    for class_index in range(num_classes):
        X_class = rng.normal(
            loc=centers[class_index],
            scale=noise_std,
            size=(num_samples_per_class, 2),
        )

        y_class = np.full(num_samples_per_class, class_index)

        X_list.append(X_class)
        y_list.append(y_class)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    permutation = rng.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    return X, y


# ------------------------------------------------------------
# 2. One-hot encoding
# ------------------------------------------------------------

def one_hot_encode(y, num_classes):
    """
    Convert integer labels into one-hot vectors.

    Example:
        y = [2, 0, 1]
        num_classes = 3

        Y =
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Integer labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    Y : np.ndarray, shape (n, num_classes)
        One-hot label matrix.
    """

    num_samples = y.shape[0]

    Y = np.zeros((num_samples, num_classes))
    Y[np.arange(num_samples), y] = 1.0

    return Y


# ------------------------------------------------------------
# 3. Softmax
# ------------------------------------------------------------

def softmax(logits):
    """
    Compute softmax probabilities row-wise.

    logits shape: (n, K)
    output shape: (n, K)

    Numerical stability trick:
        subtract max logit in each row before exponentiating.

    This does not change softmax probabilities because:

        softmax(z) = softmax(z - c)

    for any scalar c applied to all entries of one row.
    """

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probabilities


# ------------------------------------------------------------
# 4. Prediction functions
# ------------------------------------------------------------

def predict_probabilities(X, W, b):
    """
    Compute class probabilities.

    X shape: (n, d)
    W shape: (d, K)
    b shape: (K,)

    logits shape: (n, K)
    probabilities shape: (n, K)
    """

    assert X.ndim == 2
    assert W.ndim == 2
    assert b.ndim == 1
    assert X.shape[1] == W.shape[0]
    assert W.shape[1] == b.shape[0]

    logits = X @ W + b
    probabilities = softmax(logits)

    return probabilities


def predict_labels(X, W, b):
    """
    Predict class labels by choosing the class with largest probability.

    probabilities shape: (n, K)
    labels shape: (n,)
    """

    probabilities = predict_probabilities(X, W, b)
    labels = np.argmax(probabilities, axis=1)

    return labels


# ------------------------------------------------------------
# 5. Cross-entropy loss
# ------------------------------------------------------------

def cross_entropy_loss(probabilities, Y):
    """
    Multiclass cross-entropy loss.

    probabilities shape: (n, K)
    Y shape: (n, K)

    Loss:
        L = -mean over samples of log probability assigned to true class.
    """

    assert probabilities.shape == Y.shape

    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0 - eps)

    loss = -np.mean(np.sum(Y * np.log(probabilities), axis=1))

    return loss


# ------------------------------------------------------------
# 6. Compute gradients
# ------------------------------------------------------------

def compute_gradients(X, Y, W, b):
    """
    Compute gradients for softmax regression.

    errors = probabilities - Y

    grad_W = (1/n) X.T @ errors
    grad_b = (1/n) sum(errors over samples)
    """

    assert X.ndim == 2
    assert Y.ndim == 2
    assert W.ndim == 2
    assert b.ndim == 1

    num_samples = X.shape[0]

    assert Y.shape[0] == num_samples
    assert X.shape[1] == W.shape[0]
    assert W.shape[1] == Y.shape[1]
    assert b.shape[0] == Y.shape[1]

    probabilities = predict_probabilities(X, W, b)
    errors = probabilities - Y

    grad_W = (1.0 / num_samples) * (X.T @ errors)
    grad_b = (1.0 / num_samples) * np.sum(errors, axis=0)

    return grad_W, grad_b


# ------------------------------------------------------------
# 7. Accuracy
# ------------------------------------------------------------

def accuracy_score(y_pred, y_true):
    """
    Compute classification accuracy.
    """

    return np.mean(y_pred == y_true)


# ------------------------------------------------------------
# 8. Train softmax regression
# ------------------------------------------------------------

def train_softmax_regression(X, y, num_classes, learning_rate, num_iterations):
    """
    Train softmax regression using full-batch gradient descent.
    """

    num_features = X.shape[1]

    W = np.zeros((num_features, num_classes))
    b = np.zeros(num_classes)

    Y = one_hot_encode(y, num_classes)

    loss_history = []
    accuracy_history = []

    for iteration in range(num_iterations):
        probabilities = predict_probabilities(X, W, b)
        loss = cross_entropy_loss(probabilities, Y)

        y_pred = predict_labels(X, W, b)
        accuracy = accuracy_score(y_pred, y)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        grad_W, grad_b = compute_gradients(X, Y, W, b)

        W = W - learning_rate * grad_W
        b = b - learning_rate * grad_b

        if iteration % 100 == 0:
            print(
                f"Iteration {iteration:4d} | "
                f"Loss = {loss:.6f} | "
                f"Accuracy = {accuracy:.4f}"
            )

    return W, b, loss_history, accuracy_history


# ------------------------------------------------------------
# 9. Plotting
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


def plot_decision_regions(X, y, W, b):
    """
    Plot learned multiclass decision regions in 2D.
    """

    x_min = X[:, 0].min() - 1.0
    x_max = X[:, 0].max() + 1.0
    y_min = X[:, 1].min() - 1.0
    y_max = X[:, 1].max() + 1.0

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_predictions = predict_labels(grid_points, W, b)
    grid_predictions = grid_predictions.reshape(grid_x.shape)

    plt.figure()
    plt.contourf(grid_x, grid_y, grid_predictions, alpha=0.3)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")
    plt.scatter(X[y == 2, 0], X[y == 2, 1], label="Class 2")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Softmax regression decision regions")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 10. Main script
# ------------------------------------------------------------

def main():
    # Dataset settings
    num_samples_per_class = 100
    noise_std = 0.8
    num_classes = 3

    X, y = generate_multiclass_data(
        num_samples_per_class=num_samples_per_class,
        noise_std=noise_std,
        random_seed=0,
    )

    print("Data shapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nLabel values:")
    print(np.unique(y))

    Y = one_hot_encode(y, num_classes)

    print("\nOne-hot label matrix shape:")
    print("Y shape:", Y.shape)

    print("\nFirst five examples:")
    print("X[:5] =")
    print(X[:5])
    print("y[:5] =")
    print(y[:5])
    print("Y[:5] =")
    print(Y[:5])

    # Training hyperparameters
    learning_rate = 0.1
    num_iterations = 1000

    print("\nTraining softmax regression from scratch...\n")

    W_learned, b_learned, loss_history, accuracy_history = train_softmax_regression(
        X=X,
        y=y,
        num_classes=num_classes,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
    )

    print("\nLearned parameters:")
    print("W_learned shape:", W_learned.shape)
    print("W_learned:")
    print(W_learned)

    print("\nb_learned shape:", b_learned.shape)
    print("b_learned:")
    print(b_learned)

    final_probabilities = predict_probabilities(X, W_learned, b_learned)
    final_labels = predict_labels(X, W_learned, b_learned)

    final_loss = cross_entropy_loss(final_probabilities, Y)
    final_accuracy = accuracy_score(final_labels, y)

    print("\nFinal performance:")
    print("Final loss:", final_loss)
    print("Final accuracy:", final_accuracy)

    print("\nFirst five predicted probability vectors:")
    print(final_probabilities[:5])

    print("\nFirst five predicted labels:")
    print(final_labels[:5])

    print("\nFirst five true labels:")
    print(y[:5])

    print("\nProbability row sums for first five examples:")
    print(np.sum(final_probabilities[:5], axis=1))

    plot_training_curves(loss_history, accuracy_history)
    plot_decision_regions(X, y, W_learned, b_learned)


if __name__ == "__main__":
    main()