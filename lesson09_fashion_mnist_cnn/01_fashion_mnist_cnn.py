# Lesson 9: Convolutional Neural Network on Fashion-MNIST

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ------------------------------------------------------------
# 1. Device selection
# ------------------------------------------------------------

def get_device():
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
# 3. Create train/validation/test dataloaders
# ------------------------------------------------------------

def create_dataloaders(batch_size, validation_fraction=0.1, random_seed=0):
    """
    Load Fashion-MNIST.

    We split the official training set into:
        training set
        validation set

    The official test set is kept separate for final evaluation.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    full_train_dataset = datasets.FashionMNIST(
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

    num_total_train = len(full_train_dataset)
    num_val = int(validation_fraction * num_total_train)
    num_train = num_total_train - num_val

    generator = torch.Generator().manual_seed(random_seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [num_train, num_val],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# ------------------------------------------------------------
# 4. Visualize examples
# ------------------------------------------------------------

def visualize_examples(dataset, num_examples=8):
    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        image, label = dataset[i]

        # If dataset is a random_split Subset, dataset[i] still returns image, label.
        image_for_plot = image.squeeze().numpy()

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(image_for_plot, cmap="gray")
        plt.title(FASHION_MNIST_CLASSES[label], fontsize=8)
        plt.axis("off")

    plt.suptitle("Fashion-MNIST examples")
    plt.show()


# ------------------------------------------------------------
# 5. Define CNN model
# ------------------------------------------------------------

class FashionMNISTCNN(nn.Module):
    """
    Small CNN for Fashion-MNIST.

    Input:
        x shape = (batch_size, 1, 28, 28)

    Architecture:
        Conv block 1:
            Conv2d: 1 channel -> 32 channels
            ReLU
            MaxPool: 28x28 -> 14x14

        Conv block 2:
            Conv2d: 32 channels -> 64 channels
            ReLU
            MaxPool: 14x14 -> 7x7

        Classifier:
            Flatten: 64 * 7 * 7
            Linear
            ReLU
            Dropout
            Linear to 10 logits

    The model returns logits, not probabilities.
    """

    def __init__(self, num_classes=10, dropout_prob=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits


# ------------------------------------------------------------
# 6. Shape tracing helper
# ------------------------------------------------------------

def trace_model_shapes(model, device):
    """
    Pass a dummy batch through the CNN and print intermediate shapes.

    This is very useful for understanding CNN dimensions.
    """

    model.eval()

    dummy_batch = torch.zeros(4, 1, 28, 28).to(device)

    print("\nShape tracing with dummy batch:")
    print("Input:", dummy_batch.shape)

    x = dummy_batch

    for layer in model.features:
        x = layer(x)
        print(f"After {layer.__class__.__name__}:", x.shape)

    x = torch.flatten(x, start_dim=1)
    print("After Flatten:", x.shape)

    logits = model(dummy_batch)
    print("Final logits:", logits.shape)


# ------------------------------------------------------------
# 7. Evaluation
# ------------------------------------------------------------

def evaluate(model, data_loader, loss_fn, device):
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
# 8. Training loop
# ------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    learning_rate,
    weight_decay,
    num_epochs,
    device,
    checkpoint_path,
):
    """
    Train CNN and save best checkpoint according to validation accuracy.

    Important:
        We use validation accuracy for model selection.
        The test set is used only after training is complete.
    """

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0

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

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train loss = {train_loss:.4f} | "
            f"Train acc = {train_accuracy:.4f} | "
            f"Val loss = {val_loss:.4f} | "
            f"Val acc = {val_accuracy:.4f}"
        )

    print("\nBest validation accuracy:", best_val_accuracy)

    return history


# ------------------------------------------------------------
# 9. Plot training curves
# ------------------------------------------------------------

def plot_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 10. Show predictions
# ------------------------------------------------------------

def show_predictions(model, dataset, device, num_examples=10):
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
# 11. Confusion matrix and per-class accuracy
# ------------------------------------------------------------

def compute_confusion_matrix(model, data_loader, device, num_classes):
    """
    Confusion matrix:
        rows = true labels
        columns = predicted labels
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


def print_per_class_accuracy(confusion):
    print("\nPer-class accuracy:")

    for class_index, class_name in enumerate(FASHION_MNIST_CLASSES):
        correct = confusion[class_index, class_index].item()
        total = torch.sum(confusion[class_index, :]).item()
        accuracy = correct / total if total > 0 else 0.0

        print(f"{class_name:12s}: {accuracy:.4f}")


def plot_confusion_matrix(confusion):
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
    plt.title("Fashion-MNIST CNN Confusion Matrix")

    for i in range(confusion_np.shape[0]):
        for j in range(confusion_np.shape[1]):
            plt.text(
                j,
                i,
                str(confusion_np[i, j]),
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 12. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 10
    num_classes = 10
    dropout_prob = 0.3

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "fashion_mnist_cnn_best.pt",
    )

    # Data
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        batch_size=batch_size,
        validation_fraction=0.1,
        random_seed=0,
    )

    print("\nDataset sizes:")
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Test examples:", len(test_dataset))

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
    model = FashionMNISTCNN(
        num_classes=num_classes,
        dropout_prob=dropout_prob,
    ).to(device)

    print("\nModel:")
    print(model)

    trace_model_shapes(model, device)

    print("\nTraining Fashion-MNIST CNN...\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    print("\nSaved best checkpoint to:")
    print(checkpoint_path)

    plot_history(history)

    # Load best validation checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = nn.CrossEntropyLoss()

    train_loss, train_accuracy = evaluate(
        model=model,
        data_loader=train_loader,
        loss_fn=loss_fn,
        device=device,
    )

    val_loss, val_accuracy = evaluate(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nFinal evaluation using best validation checkpoint:")
    print("Best checkpoint epoch:", checkpoint["epoch"] + 1)
    print("Train loss:", train_loss)
    print("Train accuracy:", train_accuracy)
    print("Validation loss:", val_loss)
    print("Validation accuracy:", val_accuracy)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

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

    print_per_class_accuracy(confusion)
    plot_confusion_matrix(confusion)


if __name__ == "__main__":
    main()