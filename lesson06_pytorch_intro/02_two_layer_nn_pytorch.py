# Lesson 6, Part 2: Two-Layer Neural Network in PyTorch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. Device selection
# ------------------------------------------------------------

def get_device():
    """
    Select the best available device.

    cuda: NVIDIA GPU
    mps: Apple Silicon GPU
    cpu: fallback
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ------------------------------------------------------------
# 2. Generate spiral data using NumPy
# ------------------------------------------------------------

def generate_spiral_data(num_samples_per_class, num_classes, noise_std, random_seed=0):
    rng = np.random.default_rng(random_seed)

    num_samples = num_samples_per_class * num_classes
    X = np.zeros((num_samples, 2), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

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
# 3. Define a PyTorch neural network
# ------------------------------------------------------------

class TwoLayerNet(nn.Module):
    """
    Two-layer neural network:

        X -> Linear -> ReLU -> Linear -> logits

    Important:
        The model returns logits, not softmax probabilities.

    Why?
        nn.CrossEntropyLoss expects raw logits and internally applies
        log-softmax in a numerically stable way.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        # self.network = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_classes),
        # )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        logits = self.network(x)
        return logits


# ------------------------------------------------------------
# 4. Accuracy
# ------------------------------------------------------------

def accuracy_score(logits, y):
    predicted_labels = torch.argmax(logits, dim=1)
    accuracy = torch.mean((predicted_labels == y).float())
    return accuracy


# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------

def train_model(model, X_train, y_train, learning_rate, num_iterations):
    """
    Train a neural network using PyTorch autograd.

    This replaces our manual NumPy backpropagation with:

        loss.backward()
        optimizer.step()
    """

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    loss_history = []
    accuracy_history = []

    for iteration in range(num_iterations):
        # Forward pass
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        accuracy = accuracy_score(logits, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        accuracy_history.append(accuracy.item())

        if iteration % 500 == 0:
            print(
                f"Iteration {iteration:5d} | "
                f"Loss = {loss.item():.6f} | "
                f"Accuracy = {accuracy.item():.4f}"
            )

    return loss_history, accuracy_history


# ------------------------------------------------------------
# 6. Plotting
# ------------------------------------------------------------

def plot_training_curves(loss_history, accuracy_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title("PyTorch training loss")
    plt.show()

    plt.figure()
    plt.plot(accuracy_history)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("PyTorch training accuracy")
    plt.show()


def plot_decision_regions(model, X_numpy, y_numpy, device):
    x_min = X_numpy[:, 0].min() - 0.2
    x_max = X_numpy[:, 0].max() + 0.2
    y_min = X_numpy[:, 1].min() - 0.2
    y_max = X_numpy[:, 1].max() + 0.2

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()].astype(np.float32)
    grid_tensor = torch.tensor(grid_points, device=device)

    model.eval()

    with torch.no_grad():
        logits = model(grid_tensor)
        grid_predictions = torch.argmax(logits, dim=1)

    grid_predictions = grid_predictions.cpu().numpy()
    grid_predictions = grid_predictions.reshape(grid_x.shape)

    plt.figure()
    plt.contourf(grid_x, grid_y, grid_predictions, alpha=0.3)

    classes = np.unique(y_numpy)

    for class_index in classes:
        plt.scatter(
            X_numpy[y_numpy == class_index, 0],
            X_numpy[y_numpy == class_index, 1],
            label=f"Class {class_index}",
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("PyTorch two-layer neural network decision regions")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 7. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Data settings
    num_samples_per_class = 100
    num_classes = 3
    noise_std = 0.25

    X_numpy, y_numpy = generate_spiral_data(
        num_samples_per_class=num_samples_per_class,
        num_classes=num_classes,
        noise_std=noise_std,
        random_seed=0,
    )

    print("\nNumPy data:")
    print("X_numpy shape:", X_numpy.shape)
    print("y_numpy shape:", y_numpy.shape)
    print("X_numpy dtype:", X_numpy.dtype)
    print("y_numpy dtype:", y_numpy.dtype)

    # Train/test split (80% train, 20% test)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_numpy, y_numpy, test_size=0.2, random_state=42, stratify=y_numpy
    )

    print("\nTrain/test split:")
    print("X_train shape:", X_train_np.shape)
    print("X_test shape:", X_test_np.shape)
    print("y_train shape:", y_train_np.shape)
    print("y_test shape:", y_test_np.shape)

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train_np, device=device)
    y_train = torch.tensor(y_train_np, device=device)
    X_test = torch.tensor(X_test_np, device=device)
    y_test = torch.tensor(y_test_np, device=device)

    print("\nPyTorch tensors:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_train dtype:", X_train.dtype)
    print("y_train dtype:", y_train.dtype)

    # Model settings
    input_dim = 2
    hidden_dim = 64

    model = TwoLayerNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    print("\nModel:")
    print(model)

    # Training settings
    learning_rate = 0.001
    num_iterations = 5000

    print("\nTraining PyTorch two-layer neural network...\n")

    loss_history, accuracy_history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
    )

    model.eval()

    with torch.no_grad():
        # Training set performance
        train_logits = model(X_train)
        train_loss = nn.CrossEntropyLoss()(train_logits, y_train)
        train_accuracy = accuracy_score(train_logits, y_train)

        # Test set performance
        test_logits = model(X_test)
        test_loss = nn.CrossEntropyLoss()(test_logits, y_test)
        test_accuracy = accuracy_score(test_logits, y_test)

        final_probabilities = torch.softmax(train_logits, dim=1)
        final_predictions = torch.argmax(train_logits, dim=1)

    print("\nTraining set performance:")
    print(f"  Loss: {train_loss.item():.6f}")
    print(f"  Accuracy: {train_accuracy.item():.4f}")

    print("\nTest set performance:")
    print(f"  Loss: {test_loss.item():.6f}")
    print(f"  Accuracy: {test_accuracy.item():.4f}")

    print("\nFirst five probability vectors (from training set):")
    print(final_probabilities[:5].cpu().numpy())

    print("\nProbability row sums for first five examples:")
    print(torch.sum(final_probabilities[:5], dim=1).cpu().numpy())

    print("\nFirst five predicted labels:")
    print(final_predictions[:5].cpu().numpy())

    print("\nFirst five true labels:")
    print(y_train[:5].cpu().numpy())

    plot_training_curves(loss_history, accuracy_history)
    plot_decision_regions(model, X_numpy, y_numpy, device)


if __name__ == "__main__":
    main()