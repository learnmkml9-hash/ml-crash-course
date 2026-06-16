# Lesson 18: Transformer Encoder for Sequence Classification
#
# Goal:
#   Train a Transformer encoder classifier from scratch.
#
# Architecture:
#   token IDs
#       -> token embeddings
#       -> positional embeddings
#       -> prepend learned [CLS] token
#       -> Transformer encoder blocks
#       -> classifier head on [CLS]
#       -> class logits
#
# Synthetic task:
#   Given a sequence containing special_token_a and special_token_b:
#       label = 1 if special_token_a appears before special_token_b
#       label = 0 otherwise
#
# This requires positional information.

import math
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
# 2. Synthetic token-sequence dataset
# ------------------------------------------------------------

def generate_order_classification_data(
    num_examples,
    sequence_length,
    vocab_size,
    special_token_a,
    special_token_b,
    pad_token_id=0,
    min_effective_length=8,
    random_seed=0,
):
    """
    Generate token sequences for order classification.

    token_ids shape:
        (num_examples, sequence_length)

    attention_mask shape:
        (num_examples, sequence_length)

    labels shape:
        (num_examples,)

    label:
        1 if special_token_a appears before special_token_b
        0 otherwise

    We use variable effective lengths and pad the rest with pad_token_id.
    """

    assert pad_token_id == 0
    assert special_token_a != special_token_b
    assert 1 <= special_token_a < vocab_size
    assert 1 <= special_token_b < vocab_size
    assert min_effective_length >= 2
    assert min_effective_length <= sequence_length

    rng = np.random.default_rng(random_seed)

    token_ids = np.full(
        shape=(num_examples, sequence_length),
        fill_value=pad_token_id,
        dtype=np.int64,
    )

    attention_mask = np.zeros(
        shape=(num_examples, sequence_length),
        dtype=np.bool_,
    )

    labels = np.zeros(num_examples, dtype=np.int64)

    filler_tokens = [
        token
        for token in range(1, vocab_size)
        if token not in (special_token_a, special_token_b)
    ]

    for i in range(num_examples):
        effective_length = rng.integers(
            low=min_effective_length,
            high=sequence_length + 1,
        )

        # Choose two distinct positions for the special tokens.
        pos_a, pos_b = rng.choice(
            effective_length,
            size=2,
            replace=False,
        )

        label = int(pos_a < pos_b)

        sequence = rng.choice(
            filler_tokens,
            size=effective_length,
            replace=True,
        )

        sequence[pos_a] = special_token_a
        sequence[pos_b] = special_token_b

        token_ids[i, :effective_length] = sequence
        attention_mask[i, :effective_length] = True
        labels[i] = label

    return token_ids, attention_mask, labels


# ------------------------------------------------------------
# 3. Train/validation/test split
# ------------------------------------------------------------

def split_data(
    token_ids,
    attention_mask,
    labels,
    train_fraction=0.7,
    val_fraction=0.15,
    random_seed=0,
):
    rng = np.random.default_rng(random_seed)

    num_examples = token_ids.shape[0]
    indices = rng.permutation(num_examples)

    num_train = int(train_fraction * num_examples)
    num_val = int(val_fraction * num_examples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    train = (
        token_ids[train_indices],
        attention_mask[train_indices],
        labels[train_indices],
    )

    val = (
        token_ids[val_indices],
        attention_mask[val_indices],
        labels[val_indices],
    )

    test = (
        token_ids[test_indices],
        attention_mask[test_indices],
        labels[test_indices],
    )

    return train, val, test


# ------------------------------------------------------------
# 4. DataLoaders
# ------------------------------------------------------------

def create_dataloaders(train, val, test, batch_size):
    train_token_ids, train_attention_mask, train_labels = train
    val_token_ids, val_attention_mask, val_labels = val
    test_token_ids, test_attention_mask, test_labels = test

    train_dataset = TensorDataset(
        torch.tensor(train_token_ids, dtype=torch.int64),
        torch.tensor(train_attention_mask, dtype=torch.bool),
        torch.tensor(train_labels, dtype=torch.int64),
    )

    val_dataset = TensorDataset(
        torch.tensor(val_token_ids, dtype=torch.int64),
        torch.tensor(val_attention_mask, dtype=torch.bool),
        torch.tensor(val_labels, dtype=torch.int64),
    )

    test_dataset = TensorDataset(
        torch.tensor(test_token_ids, dtype=torch.int64),
        torch.tensor(test_attention_mask, dtype=torch.bool),
        torch.tensor(test_labels, dtype=torch.int64),
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

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------
# 5. Multi-head scaled dot-product attention
# ------------------------------------------------------------

def scaled_dot_product_attention_multihead(Q, K, V, mask=None):
    """
    Q, K, V:
        (B, H, T, head_dim)

    mask:
        broadcastable to (B, H, T, T)

    Returns:
        output:            (B, H, T, head_dim)
        attention_weights: (B, H, T, T)
    """

    B, H, T, head_dim = Q.shape

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights


# ------------------------------------------------------------
# 6. Multi-head self-attention
# ------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_prob):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(p=dropout_prob)

    def split_heads(self, x):
        B, T, D = x.shape

        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        B, H, T, head_dim = x.shape

        x = x.transpose(1, 2)
        x = x.contiguous().view(B, T, self.embedding_dim)

        return x

    def forward(self, x, mask=None):
        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        head_outputs, attention_weights = scaled_dot_product_attention_multihead(
            Q=Q,
            K=K,
            V=V,
            mask=mask,
        )

        head_outputs = self.dropout(head_outputs)

        concatenated = self.combine_heads(head_outputs)
        output = self.output_projection(concatenated)

        return output, attention_weights


# ------------------------------------------------------------
# 7. Feedforward network
# ------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, feedforward_dim, dropout_prob):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(feedforward_dim, embedding_dim),
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# 8. Transformer encoder block
# ------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        feedforward_dim,
        dropout_prob,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feedforward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            dropout_prob=dropout_prob,
        )

        self.residual_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, mask=None):
        normalized_x = self.norm1(x)

        attention_output, attention_weights = self.attention(
            normalized_x,
            mask=mask,
        )

        x = x + self.residual_dropout(attention_output)

        normalized_x = self.norm2(x)
        feedforward_output = self.feedforward(normalized_x)

        x = x + self.residual_dropout(feedforward_output)

        return x, attention_weights


# ------------------------------------------------------------
# 9. Transformer encoder stack
# ------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        feedforward_dim,
        num_layers,
        dropout_prob,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        all_attention_weights = []

        for layer in self.layers:
            x, attention_weights = layer(x, mask=mask)
            all_attention_weights.append(attention_weights)

        x = self.final_norm(x)

        return x, all_attention_weights


# ------------------------------------------------------------
# 10. Transformer sequence classifier
# ------------------------------------------------------------

class TransformerSequenceClassifier(nn.Module):
    """
    Transformer encoder sequence classifier.

    Inputs:
        token_ids:
            (B, T)

        attention_mask:
            (B, T), True for valid tokens and False for padding

    Internally:
        token embeddings:       (B, T, D)
        position embeddings:    (B, T, D)
        after adding [CLS]:     (B, T + 1, D)
        encoder output:         (B, T + 1, D)
        CLS representation:     (B, D)
        logits:                 (B, num_classes)
    """

    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        embedding_dim,
        num_heads,
        feedforward_dim,
        num_layers,
        num_classes,
        pad_token_id,
        dropout_prob,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

        # +1 because we prepend a [CLS] token.
        self.position_embedding = nn.Embedding(
            num_embeddings=max_sequence_length + 1,
            embedding_dim=embedding_dim,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )

        self.embedding_dropout = nn.Dropout(p=dropout_prob)

        self.encoder = TransformerEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def create_attention_mask_with_cls(self, attention_mask):
        """
        Original attention_mask:
            (B, T)

        After adding [CLS]:
            (B, T + 1)

        Transformer attention mask:
            (B, 1, 1, T + 1)

        True means valid token.
        False means padding token.
        """

        B, T = attention_mask.shape

        cls_mask = torch.ones(
            B,
            1,
            dtype=torch.bool,
            device=attention_mask.device,
        )

        mask_with_cls = torch.cat([cls_mask, attention_mask], dim=1)

        mask_with_cls = mask_with_cls.unsqueeze(1).unsqueeze(2)

        return mask_with_cls

    def forward(self, token_ids, attention_mask):
        B, T = token_ids.shape

        assert T <= self.max_sequence_length

        token_embeddings = self.token_embedding(token_ids)

        position_ids = torch.arange(
            T,
            device=token_ids.device,
        ).unsqueeze(0).expand(B, T)

        position_embeddings = self.position_embedding(position_ids + 1)

        x = token_embeddings + position_embeddings

        cls_token = self.cls_token.expand(B, 1, self.embedding_dim)

        cls_position_ids = torch.zeros(
            B,
            1,
            dtype=torch.long,
            device=token_ids.device,
        )

        cls_position_embeddings = self.position_embedding(cls_position_ids)
        cls_token = cls_token + cls_position_embeddings

        x = torch.cat([cls_token, x], dim=1)
        x = self.embedding_dropout(x)

        attention_mask_with_cls = self.create_attention_mask_with_cls(
            attention_mask=attention_mask,
        )

        encoded, all_attention_weights = self.encoder(
            x,
            mask=attention_mask_with_cls,
        )

        cls_representation = encoded[:, 0, :]

        logits = self.classifier(cls_representation)

        return logits, all_attention_weights


# ------------------------------------------------------------
# 11. Evaluation
# ------------------------------------------------------------

def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for token_ids, attention_mask, labels in data_loader:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits, _ = model(token_ids, attention_mask)
            loss = loss_fn(logits, labels)

            predictions = torch.argmax(logits, dim=1)

            batch_size = token_ids.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += torch.sum(predictions == labels).item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    average_accuracy = total_correct / total_examples

    return average_loss, average_accuracy


# ------------------------------------------------------------
# 12. Training loop
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

        for token_ids, attention_mask, labels in train_loader:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits, _ = model(token_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            predictions = torch.argmax(logits, dim=1)

            batch_size = token_ids.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += torch.sum(predictions == labels).item()
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
# 13. Plotting
# ------------------------------------------------------------

def plot_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Transformer Sequence Classifier: Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Transformer Sequence Classifier: Accuracy")
    plt.legend()
    plt.show()


def plot_attention_head(attention_weights, head_index, title):
    """
    attention_weights:
        (B, H, T + 1, T + 1)

    Plots first batch element.
    """

    attention_np = attention_weights[0, head_index].detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(attention_np)
    plt.colorbar()
    plt.xlabel("Key/value position")
    plt.ylabel("Query position")
    plt.title(title)
    plt.show()


# ------------------------------------------------------------
# 14. Inspect predictions and attention
# ------------------------------------------------------------

def inspect_predictions(
    model,
    data_loader,
    device,
    special_token_a,
    special_token_b,
    num_examples=8,
):
    model.eval()

    token_ids, attention_mask, labels = next(iter(data_loader))

    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits, all_attention_weights = model(token_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    print("\nExample predictions:")

    for i in range(num_examples):
        valid_tokens = token_ids[i][attention_mask[i]].cpu().numpy().tolist()

        pos_a = valid_tokens.index(special_token_a)
        pos_b = valid_tokens.index(special_token_b)

        print(
            f"Example {i:2d} | "
            f"pos_a = {pos_a:2d} | "
            f"pos_b = {pos_b:2d} | "
            f"true = {labels[i].item()} | "
            f"pred = {predictions[i].item()} | "
            f"prob_class_1 = {probabilities[i, 1].item():.3f} | "
            f"tokens = {valid_tokens}"
        )

    # Plot attention from last layer, head 0.
    last_layer_attention = all_attention_weights[-1]

    plot_attention_head(
        attention_weights=last_layer_attention,
        head_index=0,
        title="Last layer attention, head 0",
    )

    # Also print [CLS] attention row from head 0 for first example.
    cls_attention = last_layer_attention[0, 0, 0]

    print("\nLast layer, head 0: [CLS] attention distribution for first example:")
    print(cls_attention)


# ------------------------------------------------------------
# 15. Count parameters
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
# 16. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Dataset settings
    num_examples = 20000
    sequence_length = 20
    vocab_size = 50
    pad_token_id = 0
    special_token_a = 3
    special_token_b = 7

    token_ids, attention_mask, labels = generate_order_classification_data(
        num_examples=num_examples,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        special_token_a=special_token_a,
        special_token_b=special_token_b,
        pad_token_id=pad_token_id,
        min_effective_length=8,
        random_seed=0,
    )

    print("\nFull dataset:")
    print("token_ids shape:", token_ids.shape)
    print("attention_mask shape:", attention_mask.shape)
    print("labels shape:", labels.shape)
    print("class balance P(y=1):", np.mean(labels))

    print("\nFirst example:")
    print("token_ids[0]:", token_ids[0])
    print("attention_mask[0]:", attention_mask[0])
    print("label[0]:", labels[0])

    train, val, test = split_data(
        token_ids=token_ids,
        attention_mask=attention_mask,
        labels=labels,
        train_fraction=0.7,
        val_fraction=0.15,
        random_seed=0,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train=train,
        val=val,
        test=test,
        batch_size=128,
    )

    first_batch_token_ids, first_batch_attention_mask, first_batch_labels = next(
        iter(train_loader)
    )

    print("\nFirst batch:")
    print("token_ids shape:", first_batch_token_ids.shape)
    print("attention_mask shape:", first_batch_attention_mask.shape)
    print("labels shape:", first_batch_labels.shape)

    # Model settings
    embedding_dim = 64
    num_heads = 4
    feedforward_dim = 256
    num_layers = 3
    num_classes = 2
    dropout_prob = 0.1

    model = TransformerSequenceClassifier(
        vocab_size=vocab_size,
        max_sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        pad_token_id=pad_token_id,
        dropout_prob=dropout_prob,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("\nModel:")
    print(model)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # Quick shape trace
    model.eval()

    with torch.no_grad():
        batch_token_ids = first_batch_token_ids[:4].to(device)
        batch_attention_mask = first_batch_attention_mask[:4].to(device)

        logits, all_attention_weights = model(
            batch_token_ids,
            batch_attention_mask,
        )

    print("\nShape tracing:")
    print("logits shape:", logits.shape)
    print("number of attention tensors:", len(all_attention_weights))

    for layer_index, attention_weights in enumerate(all_attention_weights):
        print(
            f"Layer {layer_index} attention shape:",
            attention_weights.shape,
        )

    # Training settings
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 20

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "transformer_sequence_classifier_best.pt",
    )

    print("\nTraining Transformer sequence classifier...\n")

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
        special_token_a=special_token_a,
        special_token_b=special_token_b,
        num_examples=8,
    )


if __name__ == "__main__":
    main()