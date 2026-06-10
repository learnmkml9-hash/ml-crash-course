# Lesson 13: LSTM for Sequence Classification
#
# Goal:
#   Learn the PyTorch LSTM workflow:
#       sequence data
#       nn.LSTM
#       hidden state h
#       cell state c
#       sequence classification
#
# Synthetic task:
#   Input:
#       sequence of real numbers, shape (sequence_length, 1)
#
#   Label:
#       1 if the sum of the first five entries is positive
#       0 otherwise
#
# This makes the task depend on early sequence information.
# The LSTM should learn to carry this information forward through its cell state.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader


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
# 2. Generate synthetic sequence data
# ------------------------------------------------------------

def generate_sequence_data(
    num_examples,
    sequence_length,
    input_dim,
    label_rule="first_five",
    random_seed=0,
):
    """
    Generate synthetic sequence classification data.

    X shape:
        (num_examples, sequence_length, input_dim)

    y shape:
        (num_examples,)

    Label rules:
        "first_five":
            y = 1 if sum of first five time steps is positive

        "last_five":
            y = 1 if sum of last five time steps is positive
    """

    rng = np.random.default_rng(random_seed)

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_examples, sequence_length, input_dim),
    ).astype(np.float32)

    if label_rule == "first_five":
        relevant_sum = np.sum(X[:, :5, 0], axis=1)
    elif label_rule == "last_five":
        relevant_sum = np.sum(X[:, -5:, 0], axis=1)
    else:
        raise ValueError(f"Unknown label_rule: {label_rule}")

    y = (relevant_sum > 0.0).astype(np.int64)

    return X, y


# ------------------------------------------------------------
# 3. Train/validation/test split
# ------------------------------------------------------------

def split_data(X, y, train_fraction=0.7, val_fraction=0.15, random_seed=0):
    rng = np.random.default_rng(random_seed)

    num_examples = X.shape[0]
    indices = rng.permutation(num_examples)

    num_train = int(train_fraction * num_examples)
    num_val = int(val_fraction * num_examples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ------------------------------------------------------------
# 4. DataLoaders
# ------------------------------------------------------------

def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size,
):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

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

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------
# 5. LSTM classifier
# ------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    LSTM for sequence classification.

    Input:
        x shape = (batch_size, sequence_length, input_dim)

    LSTM returns:
        output, (hidden, cell)

    output shape:
        (batch_size, sequence_length, hidden_dim)

    hidden shape:
        (num_layers, batch_size, hidden_dim)

    cell shape:
        (num_layers, batch_size, hidden_dim)

    For sequence classification, we use the final hidden state
    from the last LSTM layer:

        final_hidden = hidden[-1]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout_prob=0.0,
    ):
        super().__init__()

        # PyTorch applies LSTM dropout only between LSTM layers.
        # Therefore dropout has an effect only when num_layers > 1.
        lstm_dropout = dropout_prob if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        x shape:
            (batch_size, sequence_length, input_dim)
        """

        output, (hidden, cell) = self.lstm(x)

        # Last layer's final hidden state
        final_hidden = hidden[-1]

        logits = self.classifier(final_hidden)

        return logits


# ------------------------------------------------------------
# 6. Shape tracing
# ------------------------------------------------------------

def trace_model_shapes(model, device, batch_size, sequence_length, input_dim):
    model.eval()

    dummy_batch = torch.zeros(
        batch_size,
        sequence_length,
        input_dim,
        device=device,
    )

    print("\nShape tracing:")

    print("Input x:", dummy_batch.shape)

    output, (hidden, cell) = model.lstm(dummy_batch)

    print("LSTM output:", output.shape)
    print("LSTM hidden:", hidden.shape)
    print("LSTM cell:", cell.shape)

    final_hidden = hidden[-1]
    final_cell = cell[-1]

    print("Final hidden:", final_hidden.shape)
    print("Final cell:", final_cell.shape)

    logits = model(dummy_batch)
    print("Logits:", logits.shape)


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

            # Gradient clipping is still useful for recurrent models.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    plt.title("LSTM: Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM: Accuracy")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 10. Inspect predictions
# ------------------------------------------------------------

def inspect_predictions(
    model,
    data_loader,
    device,
    label_rule,
    num_examples=10,
):
    model.eval()

    X_batch, y_batch = next(iter(data_loader))

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(X_batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    print("\nExample predictions:")

    for i in range(num_examples):
        sequence = X_batch[i, :, 0].cpu().numpy()

        if label_rule == "first_five":
            relevant_sum = np.sum(sequence[:5])
        elif label_rule == "last_five":
            relevant_sum = np.sum(sequence[-5:])
        else:
            raise ValueError(f"Unknown label_rule: {label_rule}")

        print(
            f"Example {i:2d} | "
            f"relevant_sum = {relevant_sum:+.3f} | "
            f"true = {y_batch[i].item()} | "
            f"pred = {predictions[i].item()} | "
            f"prob_class_1 = {probabilities[i, 1].item():.3f}"
        )


# ------------------------------------------------------------
# 11. Confusion matrix
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


# ------------------------------------------------------------
# 12. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Dataset settings
    num_examples = 12000
    sequence_length = 50
    input_dim = 1
    num_classes = 2

    # Try "first_five" to test longer memory.
    # Try "last_five" as an easier comparison.
    label_rule = "first_five"

    X, y = generate_sequence_data(
        num_examples=num_examples,
        sequence_length=sequence_length,
        input_dim=input_dim,
        label_rule=label_rule,
        random_seed=0,
    )

    print("\nFull dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("label rule:", label_rule)
    print("class balance P(y=1):", np.mean(y))

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X=X,
        y=y,
        train_fraction=0.7,
        val_fraction=0.15,
        random_seed=0,
    )

    print("\nDataset split:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # DataLoader settings
    batch_size = 128

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size,
    )

    first_batch_X, first_batch_y = next(iter(train_loader))

    print("\nFirst batch:")
    print("X batch shape:", first_batch_X.shape)
    print("y batch shape:", first_batch_y.shape)

    # Model settings
    hidden_dim = 64
    num_layers = 1
    dropout_prob = 0.1

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_prob=dropout_prob,
    ).to(device)

    print("\nModel:")
    print(model)

    trace_model_shapes(
        model=model,
        device=device,
        batch_size=4,
        sequence_length=sequence_length,
        input_dim=input_dim,
    )

    # Training settings
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 30

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "lstm_sequence_classifier_best.pt",
    )

    print("\nTraining LSTM...\n")

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

    inspect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        label_rule=label_rule,
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


if __name__ == "__main__":
    main()