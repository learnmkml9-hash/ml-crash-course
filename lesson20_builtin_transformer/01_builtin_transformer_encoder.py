# Lesson 20: PyTorch Built-in MultiheadAttention and TransformerEncoder
#
# Goal:
#   Connect our from-scratch Transformer implementation to PyTorch's official APIs:
#
#       nn.MultiheadAttention
#       nn.TransformerEncoderLayer
#       nn.TransformerEncoder
#
# We do two things:
#   Part A:
#       inspect nn.MultiheadAttention shapes and masks
#
#   Part B:
#       train a Transformer encoder sequence classifier using built-in modules
#
# Important PyTorch mask convention:
#   key_padding_mask:
#       shape (B, T)
#       True means "ignore this key/value position"
#
#   attn_mask:
#       shape (T, T)
#       True means "this query-key attention position is blocked"
#
# This is the opposite of our earlier from-scratch "True means allowed" mask.

import os
import math
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
# 2. Part A: Inspect nn.MultiheadAttention
# ------------------------------------------------------------

def create_causal_block_mask(sequence_length, device):
    """
    PyTorch Boolean attn_mask convention:
        True means blocked / not allowed.

    Therefore causal mask should block j > i.

    Shape:
        (T, T)
    """

    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool),
        diagonal=1,
    )

    return mask.to(device)


def demo_builtin_multihead_attention(device):
    """
    Demonstrate shapes and masks for nn.MultiheadAttention.
    """

    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 6
    embedding_dim = 16
    num_heads = 4

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device,
    )

    print("\n" + "=" * 70)
    print("Part A: nn.MultiheadAttention demo")
    print("=" * 70)

    print("\nInput:")
    print("x shape:", x.shape)

    mha = nn.MultiheadAttention(
        embed_dim=embedding_dim,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,
    ).to(device)

    print("\nBuilt-in MHA:")
    print(mha)

    # --------------------------------------------------------
    # Non-causal self-attention
    # --------------------------------------------------------

    output, weights = mha(
        query=x,
        key=x,
        value=x,
        need_weights=True,
        average_attn_weights=False,
    )

    print("\nNon-causal self-attention:")
    print("output shape:", output.shape)
    print("attention weights shape:", weights.shape)

    print("\nAttention row sums, first example, head 0:")
    print(torch.sum(weights[0, 0], dim=-1))

    # --------------------------------------------------------
    # Causal self-attention
    # --------------------------------------------------------

    causal_mask = create_causal_block_mask(
        sequence_length=sequence_length,
        device=device,
    )

    causal_output, causal_weights = mha(
        query=x,
        key=x,
        value=x,
        attn_mask=causal_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    print("\nCausal mask shape:", causal_mask.shape)
    print("Causal mask, True means blocked:")
    print(causal_mask)

    print("\nCausal self-attention:")
    print("causal_output shape:", causal_output.shape)
    print("causal attention weights shape:", causal_weights.shape)

    print("\nCausal attention, first example, head 0:")
    print(causal_weights[0, 0])

    # --------------------------------------------------------
    # Padding mask
    # --------------------------------------------------------

    token_ids = torch.tensor(
        [
            [5, 7, 9, 4, 0, 0],
            [3, 8, 1, 2, 6, 0],
        ],
        device=device,
    )

    key_padding_mask = token_ids == 0

    padded_output, padded_weights = mha(
        query=x,
        key=x,
        value=x,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    print("\nPadding mask:")
    print("token_ids:")
    print(token_ids)
    print("key_padding_mask shape:", key_padding_mask.shape)
    print("key_padding_mask, True means ignored:")
    print(key_padding_mask)

    print("\nPadding-masked attention:")
    print("padded_output shape:", padded_output.shape)
    print("padded attention weights shape:", padded_weights.shape)

    print("\nFirst example, head 0, query token 0 attention:")
    print(padded_weights[0, 0, 0])
    print("The padded key positions should receive zero attention.")

    # Plot one attention matrix.
    attention_np = causal_weights[0, 0].detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(attention_np)
    plt.colorbar()
    plt.xlabel("Key/value position")
    plt.ylabel("Query position")
    plt.title("Built-in MHA: causal attention, head 0")
    plt.show()


# ------------------------------------------------------------
# 3. Synthetic token-sequence classification data
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
    Generate variable-length token sequences.

    Label:
        1 if special_token_a appears before special_token_b
        0 otherwise

    token_ids:
        shape (N, T)

    attention_mask:
        shape (N, T)
        True means valid token.
        False means padding token.

    labels:
        shape (N,)
    """

    assert pad_token_id == 0
    assert special_token_a != special_token_b

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


def create_dataloaders(train, val, test, batch_size):
    def make_dataset(split):
        token_ids, attention_mask, labels = split

        return TensorDataset(
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
            torch.tensor(labels, dtype=torch.long),
        )

    train_loader = DataLoader(
        make_dataset(train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        make_dataset(val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        make_dataset(test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------
# 4. Built-in Transformer encoder classifier
# ------------------------------------------------------------

class BuiltinTransformerSequenceClassifier(nn.Module):
    """
    Transformer encoder classifier using PyTorch built-ins.

    Components:
        nn.Embedding
        nn.TransformerEncoderLayer
        nn.TransformerEncoder
        classifier head

    Inputs:
        token_ids:
            (B, T)

        attention_mask:
            (B, T), True for valid token, False for padding

    Internal sequence:
        prepend learned [CLS] token
        add learned positional embeddings
        pass through TransformerEncoder
        classify from CLS representation
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

        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_sequence_length + 1,
            embedding_dim=embedding_dim,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )

        self.embedding_dropout = nn.Dropout(p=dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def create_key_padding_mask_with_cls(self, attention_mask):
        """
        Input:
            attention_mask shape (B, T)
            True means valid token.

        PyTorch TransformerEncoder expects:
            src_key_padding_mask shape (B, T + 1)
            True means padding / ignored.

        Therefore we:
            1. add valid CLS mask
            2. invert mask
        """

        B, T = attention_mask.shape

        cls_valid = torch.ones(
            B,
            1,
            dtype=torch.bool,
            device=attention_mask.device,
        )

        valid_with_cls = torch.cat([cls_valid, attention_mask], dim=1)

        key_padding_mask = ~valid_with_cls

        return key_padding_mask

    def forward(self, token_ids, attention_mask):
        B, T = token_ids.shape

        assert T <= self.max_sequence_length

        token_embeddings = self.token_embedding(token_ids)

        position_ids = torch.arange(
            T,
            device=token_ids.device,
        ).unsqueeze(0).expand(B, T)

        # Positions 1..T for real tokens.
        position_embeddings = self.position_embedding(position_ids + 1)

        x = token_embeddings + position_embeddings

        # CLS token at position 0.
        cls_token = self.cls_token.expand(B, 1, self.embedding_dim)

        cls_position_ids = torch.zeros(
            B,
            1,
            dtype=torch.long,
            device=token_ids.device,
        )

        cls_token = cls_token + self.position_embedding(cls_position_ids)

        x = torch.cat([cls_token, x], dim=1)
        x = self.embedding_dropout(x)

        src_key_padding_mask = self.create_key_padding_mask_with_cls(
            attention_mask=attention_mask,
        )

        encoded = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )

        cls_representation = encoded[:, 0, :]

        logits = self.classifier(cls_representation)

        return logits


# ------------------------------------------------------------
# 5. Evaluation and training
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

            logits = model(token_ids, attention_mask)
            loss = loss_fn(logits, labels)

            predictions = torch.argmax(logits, dim=1)

            batch_size = token_ids.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += torch.sum(predictions == labels).item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


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

            logits = model(token_ids, attention_mask)
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
# 6. Plotting and inspection
# ------------------------------------------------------------

def plot_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Built-in TransformerEncoder: Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Built-in TransformerEncoder: Accuracy")
    plt.legend()
    plt.show()


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
        logits = model(token_ids, attention_mask)
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


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    return total_params, trainable_params


# ------------------------------------------------------------
# 7. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Part A: built-in MHA mechanics
    demo_builtin_multihead_attention(device=device)

    print("\n" + "=" * 70)
    print("Part B: Built-in TransformerEncoder classifier")
    print("=" * 70)

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

    train, val, test = split_data(
        token_ids=token_ids,
        attention_mask=attention_mask,
        labels=labels,
        train_fraction=0.7,
        val_fraction=0.15,
        random_seed=0,
    )

    batch_size = 128

    train_loader, val_loader, test_loader = create_dataloaders(
        train=train,
        val=val,
        test=test,
        batch_size=batch_size,
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

    model = BuiltinTransformerSequenceClassifier(
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

    # Shape trace
    model.eval()

    with torch.no_grad():
        logits = model(
            first_batch_token_ids[:4].to(device),
            first_batch_attention_mask[:4].to(device),
        )

    print("\nShape tracing:")
    print("logits shape:", logits.shape)

    # Training settings
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 20

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "builtin_transformer_encoder_classifier_best.pt",
    )

    print("\nTraining built-in TransformerEncoder classifier...\n")

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