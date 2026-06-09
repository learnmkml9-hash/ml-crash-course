# Lesson 11: Transfer Learning with Pretrained ResNet-18 on CIFAR-10

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


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
# 2. CIFAR-10 class names
# ------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ------------------------------------------------------------
# 3. Create dataloaders
# ------------------------------------------------------------

def create_dataloaders(
    batch_size,
    validation_fraction=0.1,
    random_seed=0,
    max_train_examples=None,
):
    """
    Load CIFAR-10 and create train/validation/test dataloaders.

    Important:
        ResNet-18 pretrained on ImageNet expects ImageNet-style preprocessing.
        We resize CIFAR-10 images to 224 x 224 and use ImageNet normalization.
    """

    # ImageNet normalization constants used by standard pretrained models
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_train_augmented = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    full_train_eval = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=eval_transform,
    )

    num_total = len(full_train_augmented)

    generator = torch.Generator().manual_seed(random_seed)
    all_indices = torch.randperm(num_total, generator=generator).tolist()

    if max_train_examples is not None:
        all_indices = all_indices[:max_train_examples]
        num_total = len(all_indices)

    num_val = int(validation_fraction * num_total)
    num_train = num_total - num_val

    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:]

    train_dataset = Subset(full_train_augmented, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

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
# 4. Visualization
# ------------------------------------------------------------

def unnormalize_image(image_tensor):
    """
    Undo ImageNet normalization for visualization.

    image_tensor shape:
        (3, 224, 224)
    """

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.cpu() * std + mean
    image = torch.clamp(image, 0.0, 1.0)

    return image


def visualize_examples(dataset, num_examples=8):
    plt.figure(figsize=(12, 3))

    for i in range(num_examples):
        image, label = dataset[i]

        image = unnormalize_image(image)
        image_for_plot = image.permute(1, 2, 0).numpy()

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(image_for_plot)
        plt.title(CIFAR10_CLASSES[label], fontsize=8)
        plt.axis("off")

    plt.suptitle("CIFAR-10 examples resized for ResNet-18")
    plt.show()


# ------------------------------------------------------------
# 5. Create transfer learning model
# ------------------------------------------------------------

def create_resnet18_transfer_model(num_classes, freeze_backbone=True):
    """
    Load pretrained ResNet-18 and replace the classification head.

    Original ResNet-18:
        final layer outputs 1000 ImageNet logits.

    Modified ResNet-18:
        final layer outputs num_classes logits.
    """

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    input_dim_to_fc = model.fc.in_features
    model.fc = nn.Linear(input_dim_to_fc, num_classes)

    return model


# ------------------------------------------------------------
# 6. Count trainable parameters
# ------------------------------------------------------------

def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters()
        if param.requires_grad
    )

    return total_params, trainable_params


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
    loss_fn = nn.CrossEntropyLoss()

    # Only parameters with requires_grad=True are optimized.
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
# 9. Plotting
# ------------------------------------------------------------

def plot_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Transfer Learning: Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Transfer Learning: Accuracy")
    plt.legend()
    plt.show()


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

            image_for_plot = unnormalize_image(image).permute(1, 2, 0).numpy()

            plt.subplot(2, 5, i + 1)
            plt.imshow(image_for_plot)

            title = (
                f"Pred: {CIFAR10_CLASSES[predicted_label]}\n"
                f"True: {CIFAR10_CLASSES[true_label]}\n"
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

    for class_index, class_name in enumerate(CIFAR10_CLASSES):
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
        ticks=np.arange(len(CIFAR10_CLASSES)),
        labels=CIFAR10_CLASSES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(len(CIFAR10_CLASSES)),
        labels=CIFAR10_CLASSES,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Transfer Learning Confusion Matrix")

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
# 11. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 5
    num_classes = 10

    # For faster first run, use a subset.
    # Set max_train_examples=None to use all 50,000 CIFAR-10 training examples.
    max_train_examples = 10000

    freeze_backbone = True

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "cifar10_resnet18_transfer_best.pt",
    )

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
        max_train_examples=max_train_examples,
    )

    print("\nDataset sizes:")
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Test examples:", len(test_dataset))

    first_image, first_label = train_dataset[0]

    print("\nFirst image:")
    print("image shape:", first_image.shape)
    print("label:", first_label, CIFAR10_CLASSES[first_label])
    print("image min:", torch.min(first_image).item())
    print("image max:", torch.max(first_image).item())

    first_batch_X, first_batch_y = next(iter(train_loader))

    print("\nFirst batch:")
    print("X batch shape:", first_batch_X.shape)
    print("y batch shape:", first_batch_y.shape)

    visualize_examples(train_dataset, num_examples=8)

    print("\nLoading pretrained ResNet-18...")

    model = create_resnet18_transfer_model(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("\nModel:")
    print(model)

    print("\nParameter counts:")
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    print("\nTraining transfer-learning model...\n")

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