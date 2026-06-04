# Lesson 2: Linear Regression from Scratch Using NumPy

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Generate synthetic linear regression data
# ------------------------------------------------------------

def generate_data(num_samples, num_features, noise_std, random_seed=0):
    """
    Generate synthetic data from a linear model:

        y = X w_true + b_true + noise

    Parameters
    ----------
    num_samples : int
        Number of data points, usually denoted n.
    num_features : int
        Number of features, usually denoted d.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (num_samples, num_features)
        Feature matrix.
    y : np.ndarray, shape (num_samples,)
        Target vector.
    w_true : np.ndarray, shape (num_features,)
        Ground-truth weight vector.
    b_true : float
        Ground-truth bias.
    """

    rng = np.random.default_rng(random_seed)

    X = rng.normal(size=(num_samples, num_features))

    w_true = rng.normal(size=num_features)
    b_true = 2.0

    noise = noise_std * rng.normal(size=num_samples)

    y = X @ w_true + b_true + noise

    return X, y, w_true, b_true


# ------------------------------------------------------------
# 2. Define the model
# ------------------------------------------------------------

def predict(X, w, b):
    """
    Linear model:

        y_hat = X w + b

    X shape: (n, d)
    w shape: (d,)
    b shape: scalar

    output shape: (n,)
    """

    return X @ w + b


# ------------------------------------------------------------
# 3. Define the loss function
# ------------------------------------------------------------

def mean_squared_error(y_pred, y_true):
    """
    Mean squared error:

        L = mean((y_pred - y_true)^2)
    """

    errors = y_pred - y_true
    loss = np.mean(errors ** 2)

    return loss


# ------------------------------------------------------------
# 4. Compute gradients
# ------------------------------------------------------------

def compute_gradients(X, y, w, b):
    """
    Compute gradients of the mean squared error loss.

    Loss:

        L(w, b) = (1/n) sum_i (x_i^T w + b - y_i)^2

    Gradients:

        grad_w = (2/n) X^T (Xw + b - y)
        grad_b = (2/n) sum_i (x_i^T w + b - y_i)
    """

    num_samples = X.shape[0]

    y_pred = predict(X, w, b)
    errors = y_pred - y

    grad_w = (2.0 / num_samples) * (X.T @ errors)
    grad_b = (2.0 / num_samples) * np.sum(errors)

    return grad_w, grad_b

def closed_form_solution(X, y):
    """
    Compute the ordinary least squares solution using the normal equation.

    We augment X with a column of ones to absorb the bias term.
    """

    num_samples = X.shape[0]
    ones = np.ones((num_samples, 1))
    X_augmented = np.hstack([X, ones])

    theta = np.linalg.solve(X_augmented.T @ X_augmented, X_augmented.T @ y)

    w_closed_form = theta[:-1]
    b_closed_form = theta[-1]

    return w_closed_form, b_closed_form


# ------------------------------------------------------------
# 5. Train with gradient descent
# ------------------------------------------------------------

def train_linear_regression(X, y, learning_rate, num_iterations):
    """
    Train linear regression using gradient descent.

    Returns loss history and gradient norm history.
    """

    num_features = X.shape[1]

    # Initialize parameters
    w = np.zeros(num_features)
    b = 0.0

    loss_history = []
    grad_norm_history = []

    for iteration in range(num_iterations):
        # Forward pass
        y_pred = predict(X, w, b)
        loss = mean_squared_error(y_pred, y)

        # Backward pass: compute gradients
        grad_w, grad_b = compute_gradients(X, y, w, b)
        grad_norm = np.linalg.norm(np.append(grad_w, grad_b))

        # Store metrics for analysis
        loss_history.append(loss)
        grad_norm_history.append(grad_norm)

        # Gradient descent update
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        # Print progress occasionally
        if iteration % 100 == 0:
            print(
                f"Iteration {iteration:4d} | Loss = {loss:.6f} | "
                f"Grad norm = {grad_norm:.6f}"
            )

    return w, b, loss_history, grad_norm_history


# ------------------------------------------------------------
# 6. Main script
# ------------------------------------------------------------

def main():
    # Problem dimensions
    num_samples = 200
    num_features = 3
    noise_std = 0.5

    # Generate data
    X, y, w_true, b_true = generate_data(
        num_samples=num_samples,
        num_features=num_features,
        noise_std=noise_std,
        random_seed=0,
    )

    print("Data shapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("w_true shape:", w_true.shape)
    print("b_true:", b_true)

    print("\nGround-truth parameters:")
    print("w_true:", w_true)
    print("b_true:", b_true)

    # Training hyperparameters
    learning_rate = 0.01
    num_iterations = 1000

    print("\nTraining linear regression from scratch...\n")

    w_learned, b_learned, loss_history, grad_norm_history = train_linear_regression(
        X=X,
        y=y,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
    )

    print("\nLearned parameters:")
    print("w_learned:", w_learned)
    print("b_learned:", b_learned)

    print("\nParameter error:")
    print("w_learned - w_true:", w_learned - w_true)
    print("b_learned - b_true:", b_learned - b_true)

    final_predictions = predict(X, w_learned, b_learned)
    final_loss = mean_squared_error(final_predictions, y)

    print("\nFinal training loss:")
    print(final_loss)

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Mean squared error")
    plt.title("Gradient descent training curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot gradient norm curve
    plt.figure(figsize=(8, 4))
    plt.plot(grad_norm_history, label="Gradient norm", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient norm")
    plt.title("Gradient norm during training")
    plt.legend()
    plt.tight_layout()
    plt.show()

    w_closed_form, b_closed_form = closed_form_solution(X, y)

    print("\nClosed-form least-squares parameters:")
    print("w_closed_form:", w_closed_form)
    print("b_closed_form:", b_closed_form)

    print("\nDifference between gradient descent and closed form:")
    print("w_learned - w_closed_form:", w_learned - w_closed_form)
    print("b_learned - b_closed_form:", b_learned - b_closed_form)


if __name__ == "__main__":
    main()