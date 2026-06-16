# Lesson 14: GRU and Direct RNN/LSTM/GRU Comparison
#
# Goal:
#   Compare Vanilla RNN, LSTM, and GRU on the same sequence classification task.
#
# Synthetic long-memory task:
#   Input:
#       sequence of real numbers, shape (sequence_length, 1)
#
#   Label:
#       1 if the sum of the first five entries is positive
#       0 otherwise
#
# Because the relevant information appears at the beginning of the sequence,
# the model must remember it until the final classification step.

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
# 2. Synthetic sequence data
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

    label_rule:
        "first_five":
            label depends on sum of first five time steps

        "last_five":
            label depends on sum of last five time steps
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
# 5. Unified recurrent classifier
# ------------------------------------------------------------

class RecurrentClassifier(nn.Module):
    """
    Unified sequence classifier supporting:
        - vanilla RNN
        - LSTM
        - GRU

    Input:
        x shape = (batch_size, sequence_length, input_dim)

    Output:
        logits shape = (batch_size, num_classes)
    """

    def __init__(
        self,
        model_type,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout_prob=0.0,
        rnn_nonlinearity="tanh",
    ):
        super().__init__()

        self.model_type = model_type.lower()

        recurrent_dropout = dropout_prob if num_layers > 1 else 0.0

        if self.model_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                nonlinearity=rnn_nonlinearity,
                batch_first=True,
                dropout=recurrent_dropout,
            )

        elif self.model_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )

        elif self.model_type == "gru":
            self.recurrent = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        For RNN and GRU:
            output, hidden = recurrent(x)

        For LSTM:
            output, (hidden, cell) = recurrent(x)

        In all cases:
            hidden shape = (num_layers, batch_size, hidden_dim)

        We use hidden[-1], the final hidden state of the last recurrent layer.
        """

        if self.model_type == "lstm":
            output, (hidden, cell) = self.recurrent(x)
        else:
            output, hidden = self.recurrent(x)

        final_hidden = hidden[-1]
        logits = self.classifier(final_hidden)

        return logits


# ------------------------------------------------------------
# 6. Shape tracing
# ------------------------------------------------------------

def trace_model_shapes(
    model,
    model_name,
    device,
    batch_size,
    sequence_length,
    input_dim,
):
    model.eval()

    dummy_batch = torch.zeros(
        batch_size,
        sequence_length,
        input_dim,
        device=device,
    )

    print(f"\nShape tracing for {model_name}:")
    print("Input x:", dummy_batch.shape)

    if model.model_type == "lstm":
        output, (hidden, cell) = model.recurrent(dummy_batch)

        print("Recurrent output:", output.shape)
        print("Hidden:", hidden.shape)
        print("Cell:", cell.shape)

    else:
        output, hidden = model.recurrent(dummy_batch)

        print("Recurrent output:", output.shape)
        print("Hidden:", hidden.shape)

    final_hidden = hidden[-1]
    print("Final hidden:", final_hidden.shape)

    logits = model(dummy_batch)
    print("Logits:", logits.shape)


# ------------------------------------------------------------
# 7. Parameter counting
# ------------------------------------------------------------

def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    return total_params, trainable_params


# ------------------------------------------------------------
# 8. Evaluation
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
# 9. Training one model
# ------------------------------------------------------------

def train_one_model(
    model,
    model_name,
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

    print(f"\nTraining {model_name}...\n")

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

            # Important for recurrent models.
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
            f"{model_name:4s} | "
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train loss = {train_loss:.4f} | "
            f"Train acc = {train_accuracy:.4f} | "
            f"Val loss = {val_loss:.4f} | "
            f"Val acc = {val_accuracy:.4f}"
        )

    print(f"\nBest validation accuracy for {model_name}: {best_val_accuracy:.4f}")

    return history


# ------------------------------------------------------------
# 10. Plot comparison
# ------------------------------------------------------------

def plot_comparison(all_histories):
    """
    Plot validation loss and validation accuracy for all models.
    """

    plt.figure()

    for model_name, history in all_histories.items():
        plt.plot(history["val_loss"], label=f"{model_name} val loss")

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("RNN vs LSTM vs GRU: Validation Loss")
    plt.legend()
    plt.show()

    plt.figure()

    for model_name, history in all_histories.items():
        plt.plot(history["val_accuracy"], label=f"{model_name} val accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("RNN vs LSTM vs GRU: Validation Accuracy")
    plt.legend()
    plt.show()


# ------------------------------------------------------------
# 11. Inspect predictions
# ------------------------------------------------------------

def inspect_predictions(
    model,
    model_name,
    data_loader,
    device,
    label_rule,
    num_examples=5,
):
    model.eval()

    X_batch, y_batch = next(iter(data_loader))

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(X_batch)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    print(f"\nExample predictions for {model_name}:")

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
# 12. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Dataset settings
    num_examples = 12000
    sequence_length = 80
    input_dim = 1
    num_classes = 2

    # "first_five" is the long-memory task.
    # "last_five" is easier for all recurrent models.
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

    # Shared model/training settings
    hidden_dim = 64
    num_layers = 1
    dropout_prob = 0.1

    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 20

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_types = ["rnn", "lstm", "gru"]

    all_histories = {}
    final_results = {}

    for model_type in model_types:
        model_name = model_type.upper()

        # Reset seed before creating each model for reproducibility.
        torch.manual_seed(0)

        model = RecurrentClassifier(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            rnn_nonlinearity="tanh",
        ).to(device)

        total_params, trainable_params = count_parameters(model)

        print("\n" + "=" * 70)
        print(f"Model: {model_name}")
        print(model)
        print("Total parameters:", total_params)
        print("Trainable parameters:", trainable_params)

        trace_model_shapes(
            model=model,
            model_name=model_name,
            device=device,
            batch_size=4,
            sequence_length=sequence_length,
            input_dim=input_dim,
        )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{model_type}_sequence_classifier_best.pt",
        )

        history = train_one_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            device=device,
            checkpoint_path=checkpoint_path,
        )

        all_histories[model_name] = history

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

        final_results[model_name] = {
            "best_epoch": checkpoint["epoch"] + 1,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

        inspect_predictions(
            model=model,
            model_name=model_name,
            data_loader=test_loader,
            device=device,
            label_rule=label_rule,
            num_examples=5,
        )

    # --------------------------------------------------------
    # Final comparison table
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Final comparison using best validation checkpoints")
    print("=" * 70)

    header = (
        f"{'Model':8s} | "
        f"{'BestEp':>6s} | "
        f"{'Params':>8s} | "
        f"{'TrainAcc':>8s} | "
        f"{'ValAcc':>8s} | "
        f"{'TestAcc':>8s} | "
        f"{'TestLoss':>8s}"
    )

    print(header)
    print("-" * len(header))

    for model_name, result in final_results.items():
        print(
            f"{model_name:8s} | "
            f"{result['best_epoch']:6d} | "
            f"{result['trainable_params']:8d} | "
            f"{result['train_accuracy']:8.4f} | "
            f"{result['val_accuracy']:8.4f} | "
            f"{result['test_accuracy']:8.4f} | "
            f"{result['test_loss']:8.4f}"
        )

    plot_comparison(all_histories)


if __name__ == "__main__":
    main()