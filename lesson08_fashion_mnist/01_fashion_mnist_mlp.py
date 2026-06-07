# Lesson 8: Training on Fashion-MNIST with PyTorch
#
# Goal:
#   Train a real image classifier using:
#       - torchvision.datasets.FashionMNIST
#       - transforms
#       - DataLoader
#       - an MLP neural network
#       - train/test evaluation
#       - checkpointing

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------
# 1. Device selection
# ------------------------------------------------------------

def get_device():
    """
    Select best available device.

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
# 2. Class names
# ------------------------------------------------------------

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ------------------------------------------------------------
# 3. Create datasets and dataloaders
# ------------------------------------------------------------

def create_dataloaders(batch_size):
    """
    Load Fashion-MNIST train and test datasets.

    transform:
        ToTensor converts a PIL image into a tensor with shape (1, 28, 28)
        and values in [0, 1].

        Normalize approximately centers and scales pixel values.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

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

    return train_dataset, test_dataset, train_loader, test_loader


# ------------------------------------------------------------
# 4. Visualize a few examples
# ------------------------------------------------------------

def visualize_examples(dataset, num_examples=8):
    """
    Show a few Fashion-MNIST images.

    Each image has shape (1, 28, 28).
    For plotting, we squeeze it to shape (28, 28).
    """

    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        image, label = dataset[i]

        image_for_plot = image.squeeze().numpy()

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(image_for_plot, cmap="gray")
        plt.title(FASHION_MNIST_CLASSES[label])
        plt.axis("off")

    plt.suptitle("Fashion-MNIST examples")
    plt.show()


# ------------------------------------------------------------
# 5. Define MLP model
# ------------------------------------------------------------

class FashionMNISTMLP(nn.Module):
    """
    Multilayer perceptron for Fashion-MNIST.

    Input image shape:
        (batch_size, 1, 28, 28)

    We flatten each image:
        1 * 28 * 28 = 784

    Then apply:
        Linear -> ReLU -> Linear -> ReLU -> Linear

    The model returns logits, not probabilities.
    """

    def __init__(self, hidden_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        logits = self.network(x)
        return logits


# ------------------------------------------------------------
# 6. Evaluation helper
# ------------------------------------------------------------

def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluate model on a data loader.

    Returns:
        average loss
        average accuracy
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

            predictions = torch.argmax(logits, dim=1)

            batch_size = X_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += torch.sum(predictions == y_batch).item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples

    return average_loss, average_accuracy


# ------------------------------------------------------------
# 7. Training loop
# ------------------------------------------------------------

def train_model(
    model,
    train_loader,
    test_loader,
    learning_rate,
    num_epochs,
    device,
    checkpoint_path,
):
    """
    Train the model and save the best checkpoint according to test accuracy.
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    best_test_accuracy = 0.0

    for epoch in range(num_epochs):
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

            predictions = torch.argmax(logits, dim=1)

            batch_size = X_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += torch.sum(predictions == y_batch).item()
            total_examples += batch_size

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples

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

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_accuracy": test_accuracy,
                    "test_loss": test_loss,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train loss = {train_loss:.4f} | "
            f"Train acc = {train_accuracy:.4f} | "
            f"Test loss = {test_loss:.4f} | "
            f"Test acc = {test_accuracy:.4f}"
        )

    print("\nBest test accuracy:", best_test_accuracy)

    return history


# ------------------------------------------------------------
# 8. Plot training curves
# ------------------------------------------------------------

def plot_history(history):
    """
    Plot train/test loss and accuracy.
    """

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["test_loss"], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Fashion-MNIST MLP: Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["test_accuracy"], label="test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Fashion-MNIST MLP: Accuracy")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 9. Show predictions
# ------------------------------------------------------------

def show_predictions(model, dataset, device, num_examples=10):
    """
    Show model predictions on a few test images.
    """

    model.eval()

    plt.figure(figsize=(15, 4))

    with torch.no_grad():
        for i in range(num_examples):
            image, true_label = dataset[i]

            image_batch = image.unsqueeze(0).to(device)
            logits = model(image_batch)
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            image_for_plot = image.squeeze().numpy()

            plt.subplot(2, 5, i + 1)
            plt.imshow(image_for_plot, cmap="gray")

            title = (
                f"Pred: {FASHION_MNIST_CLASSES[predicted_label]}\n"
                f"True: {FASHION_MNIST_CLASSES[true_label]}\n"
                f"Conf: {confidence:.2f}"
            )

            plt.title(title, fontsize=8)
            plt.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 10. Confusion matrix
# ------------------------------------------------------------

def compute_confusion_matrix(model, data_loader, device, num_classes):
    """
    Compute confusion matrix without scikit-learn.

    rows: true labels
    columns: predicted labels
    """

    model.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            predictions = torch.argmax(logits, dim=1)

            for true_label, predicted_label in zip(y_batch, predictions):
                confusion[true_label.item(), predicted_label.item()] += 1

    return confusion


def plot_confusion_matrix(confusion):
    """
    Plot confusion matrix.
    """

    confusion_np = confusion.numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_np)
    plt.colorbar()

    plt.xticks(
        ticks=np.arange(len(FASHION_MNIST_CLASSES)),
        labels=FASHION_MNIST_CLASSES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(len(FASHION_MNIST_CLASSES)),
        labels=FASHION_MNIST_CLASSES,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Fashion-MNIST confusion matrix")

    for i in range(confusion_np.shape[0]):
        for j in range(confusion_np.shape[1]):
            plt.text(j, i, str(confusion_np[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Hyperparameters
    batch_size = 128
    hidden_dim = 64
    num_classes = 10
    learning_rate = 1e-3
    num_epochs = 20

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "fashion_mnist_mlp_best.pt",
    )

    # Data
    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
    )

    print("\nDataset sizes:")
    print("Number of training examples:", len(train_dataset))
    print("Number of test examples:", len(test_dataset))

    first_image, first_label = train_dataset[0]
    print("\nFirst image:")
    print("image shape:", first_image.shape)
    print("label:", first_label, FASHION_MNIST_CLASSES[first_label])
    print("image min:", torch.min(first_image).item())
    print("image max:", torch.max(first_image).item())

    first_batch_X, first_batch_y = next(iter(train_loader))
    print("\nFirst batch:")
    print("X batch shape:", first_batch_X.shape)
    print("y batch shape:", first_batch_y.shape)

    visualize_examples(train_dataset, num_examples=8)

    # Model
    model = FashionMNISTMLP(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    print("\nModel:")
    print(model)

    # Train
    print("\nTraining Fashion-MNIST MLP...\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    print("\nSaved best checkpoint to:")
    print(checkpoint_path)

    # Plot learning curves
    plot_history(history)

    # Load best model before final evaluation
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = nn.CrossEntropyLoss()

    final_train_loss, final_train_accuracy = evaluate(
        model=model,
        data_loader=train_loader,
        loss_fn=loss_fn,
        device=device,
    )

    final_test_loss, final_test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nFinal evaluation using best checkpoint:")
    print("Best checkpoint epoch:", checkpoint["epoch"] + 1)
    print("Final train loss:", final_train_loss)
    print("Final train accuracy:", final_train_accuracy)
    print("Final test loss:", final_test_loss)
    print("Final test accuracy:", final_test_accuracy)

    show_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        num_examples=10,
    )

    confusion = compute_confusion_matrix(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=num_classes,
    )

    print("\nConfusion matrix:")
    print(confusion)

    plot_confusion_matrix(confusion)


if __name__ == "__main__":
    main()