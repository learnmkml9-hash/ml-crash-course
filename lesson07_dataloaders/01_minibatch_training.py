# Lesson 7: Dataset, DataLoader, Mini-Batch Training, and Train/Test Split

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Device selection
# ------------------------------------------------------------

def get_device():
    """
    Select best available device:
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
    """
    Generate a nonlinear 2D spiral dataset.

    X shape: (n, 2)
    y shape: (n,)
    """

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
# 3. Train/test split
# ------------------------------------------------------------

def train_test_split_numpy(X, y, train_fraction=0.8, random_seed=0):
    """
    Split NumPy arrays into train and test sets.

    We do this manually instead of using scikit-learn, so that the mechanics
    are transparent.
    """

    rng = np.random.default_rng(random_seed)

    num_samples = X.shape[0]
    indices = rng.permutation(num_samples)

    num_train = int(train_fraction * num_samples)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------
# 4. Create DataLoaders
# ------------------------------------------------------------

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    """
    Convert NumPy arrays into PyTorch TensorDatasets and DataLoaders.

    Important:
        We keep the dataset tensors on CPU.
        During training, each mini-batch is moved to the selected device.
    """

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


# ------------------------------------------------------------
# 5. Define model
# ------------------------------------------------------------

class TwoLayerNet(nn.Module):
    """
    Two-layer neural network:

        x -> Linear -> ReLU -> Linear -> logits

    The model returns logits, not probabilities.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        logits = self.network(x)
        return logits


# ------------------------------------------------------------
# 6. Accuracy helper
# ------------------------------------------------------------

def accuracy_from_logits(logits, y):
    """
    Compute accuracy from logits and integer labels.
    """

    predictions = torch.argmax(logits, dim=1)
    accuracy = torch.mean((predictions == y).float())

    return accuracy


# ------------------------------------------------------------
# 7. One training epoch
# ------------------------------------------------------------

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    Train for one pass over the training dataset.

    Mini-batch training loop:

        for each batch:
            move batch to device
            forward pass
            compute loss
            clear gradients
            backward pass
            update parameters
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X_batch.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples

    return average_loss, average_accuracy


# ------------------------------------------------------------
# 8. Evaluation
# ------------------------------------------------------------

def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluate model on a dataset.

    No gradients are needed during evaluation.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            batch_size = X_batch.shape[0]

            total_loss += loss.item() * batch_size
            total_correct += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples

    return average_loss, average_accuracy


# ------------------------------------------------------------
# 9. Full training loop
# ------------------------------------------------------------

def train_model(
    model,
    train_loader,
    test_loader,
    learning_rate,
    num_epochs,
    device,
):
    """
    Train model using mini-batch gradient descent.
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Train loss = {train_loss:.6f} | "
                f"Train acc = {train_accuracy:.4f} | "
                f"Test loss = {test_loss:.6f} | "
                f"Test acc = {test_accuracy:.4f}"
            )

    return history


# ------------------------------------------------------------
# 10. Plotting
# ------------------------------------------------------------

def plot_training_history(history):
    """
    Plot train/test loss and train/test accuracy.
    """

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["test_loss"], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Train/Test Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["test_accuracy"], label="test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy")
    plt.legend()
    plt.show()


def plot_decision_regions(model, X_numpy, y_numpy, device):
    """
    Plot learned decision regions.
    """

    x_min = X_numpy[:, 0].min() - 0.2
    x_max = X_numpy[:, 0].max() + 0.2
    y_min = X_numpy[:, 1].min() - 0.2
    y_max = X_numpy[:, 1].max() + 0.2

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )

    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()].astype(np.float32)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(grid_tensor)
        predictions = torch.argmax(logits, dim=1)

    predictions = predictions.cpu().numpy()
    predictions = predictions.reshape(grid_x.shape)

    plt.figure()
    plt.contourf(grid_x, grid_y, predictions, alpha=0.3)

    for class_index in np.unique(y_numpy):
        plt.scatter(
            X_numpy[y_numpy == class_index, 0],
            X_numpy[y_numpy == class_index, 1],
            label=f"Class {class_index}",
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision regions with mini-batch training")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 11. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Dataset settings
    num_samples_per_class = 200
    num_classes = 3
    noise_std = 0.35

    X, y = generate_spiral_data(
        num_samples_per_class=num_samples_per_class,
        num_classes=num_classes,
        noise_std=noise_std,
        random_seed=0,
    )

    print("\nFull dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, y_train, X_test, y_test = train_test_split_numpy(
        X=X,
        y=y,
        train_fraction=0.8,
        random_seed=0,
    )

    print("\nTrain/test split:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # DataLoader settings
    batch_size = 32

    train_loader, test_loader = create_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
    )

    print("\nDataLoader:")
    print("Number of train batches:", len(train_loader))
    print("Number of test batches:", len(test_loader))

    first_batch_X, first_batch_y = next(iter(train_loader))

    print("\nFirst training batch:")
    print("X batch shape:", first_batch_X.shape)
    print("y batch shape:", first_batch_y.shape)

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
    learning_rate = 0.01
    num_epochs = 200

    print("\nTraining with mini-batches...\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
    )

    final_train_loss = history["train_loss"][-1]
    final_train_accuracy = history["train_accuracy"][-1]
    final_test_loss = history["test_loss"][-1]
    final_test_accuracy = history["test_accuracy"][-1]

    print("\nFinal performance:")
    print("Final train loss:", final_train_loss)
    print("Final train accuracy:", final_train_accuracy)
    print("Final test loss:", final_test_loss)
    print("Final test accuracy:", final_test_accuracy)

    plot_training_history(history)
    plot_decision_regions(model, X, y, device)


if __name__ == "__main__":
    main()