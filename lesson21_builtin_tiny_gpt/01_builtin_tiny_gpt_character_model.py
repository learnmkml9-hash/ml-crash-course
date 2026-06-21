# Lesson 21: Tiny GPT Using PyTorch Built-in Transformer Modules
#
# Goal:
#   Rebuild the tiny GPT-style character language model from Lesson 19,
#   but using PyTorch's built-in TransformerEncoderLayer and TransformerEncoder.
#
# Important idea:
#   A TransformerEncoder stack with a causal mask can be used as a
#   decoder-only GPT-style model for next-token prediction.
#
# Architecture:
#   token ids
#       -> token embeddings
#       -> positional embeddings
#       -> TransformerEncoder with causal mask
#       -> final LayerNorm
#       -> LM head
#       -> logits over vocabulary
#
# Training objective:
#   next-token prediction

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
# 2. Tiny text corpus
# ------------------------------------------------------------

def build_tiny_corpus():
    """
    Build a small repeated character-level corpus.

    This is intentionally tiny so training is fast and interpretable.
    """

    base_text = """
    the cat sat on the mat.
    the dog sat on the log.
    the bird flew over the tree.
    the fish swam in the pond.
    the cat chased the bird.
    the dog chased the cat.
    the bird saw the dog.
    the fish saw the cat.
    """

    text = base_text.lower() * 200
    text = " ".join(text.split())

    return text


# ------------------------------------------------------------
# 3. Character tokenizer
# ------------------------------------------------------------

class CharacterTokenizer:
    """
    Minimal character-level tokenizer.
    """

    def __init__(self, text):
        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids):
        return "".join(self.itos[int(i)] for i in token_ids)


# ------------------------------------------------------------
# 4. Train/validation split
# ------------------------------------------------------------

def create_train_val_data(token_ids, train_fraction=0.9):
    data = torch.tensor(token_ids, dtype=torch.long)

    num_train = int(train_fraction * len(data))

    train_data = data[:num_train]
    val_data = data[num_train:]

    return train_data, val_data


# ------------------------------------------------------------
# 5. Batch sampling
# ------------------------------------------------------------

def get_batch(data, batch_size, block_size, device):
    """
    Sample random contiguous chunks.

    x shape:
        (B, T)

    y shape:
        (B, T)

    y is x shifted one step into the future.
    """

    max_start = len(data) - block_size - 1

    start_indices = torch.randint(
        low=0,
        high=max_start,
        size=(batch_size,),
    )

    x = torch.stack([
        data[start:start + block_size]
        for start in start_indices
    ])

    y = torch.stack([
        data[start + 1:start + block_size + 1]
        for start in start_indices
    ])

    return x.to(device), y.to(device)


# ------------------------------------------------------------
# 6. Causal mask for built-in Transformer
# ------------------------------------------------------------

def create_causal_block_mask(sequence_length, device):
    """
    Create Boolean causal attention mask for PyTorch Transformer modules.

    PyTorch Boolean attention-mask convention:
        True  = blocked / not allowed to attend
        False = allowed

    For causal language modeling:
        position i may not attend to positions j > i.

    Shape:
        (T, T)
    """

    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool),
        diagonal=1,
    )

    return mask.to(device)


# ------------------------------------------------------------
# 7. Built-in Tiny GPT model
# ------------------------------------------------------------

class BuiltinTinyGPT(nn.Module):
    """
    Tiny GPT-style language model using PyTorch built-in TransformerEncoder.

    Input:
        token_ids shape = (B, T)

    Output:
        logits shape = (B, T, vocab_size)

    Each position predicts the next token.
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim,
        num_heads,
        feedforward_dim,
        num_layers,
        dropout_prob,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim,
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

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, targets=None):
        """
        token_ids:
            (B, T)

        targets:
            (B, T), optional
        """

        B, T = token_ids.shape

        assert T <= self.block_size

        position_ids = torch.arange(
            T,
            device=token_ids.device,
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings
        x = self.embedding_dropout(x)

        causal_mask = create_causal_block_mask(
            sequence_length=T,
            device=token_ids.device,
        )

        # Built-in TransformerEncoder normally allows bidirectional attention.
        # Passing this causal mask makes it decoder-only / GPT-style.
        x = self.transformer(
            x,
            mask=causal_mask,
        )

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            logits_flat = logits.reshape(B * T, self.vocab_size)
            targets_flat = targets.reshape(B * T)

            loss = nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
            )

        return logits, loss


# ------------------------------------------------------------
# 8. Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    device,
    eval_iters,
):
    model.eval()

    results = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(eval_iters):
            x_batch, y_batch = get_batch(
                data=data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
            )

            _, loss = model(x_batch, targets=y_batch)
            losses.append(loss.item())

        results[split_name] = sum(losses) / len(losses)

    model.train()

    return results


# ------------------------------------------------------------
# 9. Text generation
# ------------------------------------------------------------

@torch.no_grad()
def generate(model, context, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressive text generation.

    context:
        token ids, shape (B, T)
    """

    model.eval()

    for _ in range(max_new_tokens):
        context_cropped = context[:, -model.block_size:]

        logits, _ = model(context_cropped)

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature

        if top_k is not None:
            values, indices = torch.topk(next_token_logits, k=top_k)

            filtered_logits = torch.full_like(
                next_token_logits,
                fill_value=float("-inf"),
            )

            filtered_logits.scatter_(
                dim=1,
                index=indices,
                src=values,
            )

            next_token_logits = filtered_logits

        probabilities = torch.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        context = torch.cat([context, next_token], dim=1)

    model.train()

    return context


# ------------------------------------------------------------
# 10. Mask inspection
# ------------------------------------------------------------

def inspect_causal_mask(block_size, device):
    mask = create_causal_block_mask(
        sequence_length=block_size,
        device=device,
    )

    print("\nCausal mask:")
    print("mask shape:", mask.shape)
    print("True means blocked.")
    print(mask)


# ------------------------------------------------------------
# 11. Count parameters
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
# 12. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    text = build_tiny_corpus()

    tokenizer = CharacterTokenizer(text)

    token_ids = tokenizer.encode(text)

    train_data, val_data = create_train_val_data(
        token_ids=token_ids,
        train_fraction=0.9,
    )

    print("\nCorpus:")
    print("Number of characters:", len(text))
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Vocabulary:", "".join(tokenizer.itos[i] for i in range(tokenizer.vocab_size)))
    print("First 200 chars:")
    print(text[:200])

    print("\nData:")
    print("train_data length:", len(train_data))
    print("val_data length:", len(val_data))

    # Model hyperparameters
    block_size = 32
    batch_size = 64
    embedding_dim = 64
    num_heads = 4
    feedforward_dim = 4 * embedding_dim
    num_layers = 3
    dropout_prob = 0.1

    # Training hyperparameters
    learning_rate = 3e-4
    weight_decay = 1e-4
    max_steps = 2000
    eval_interval = 200
    eval_iters = 50

    inspect_causal_mask(
        block_size=8,
        device=device,
    )

    model = BuiltinTinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    print("\nModel:")
    print(model)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # Shape tracing
    x_batch, y_batch = get_batch(
        data=train_data,
        batch_size=4,
        block_size=block_size,
        device=device,
    )

    logits, loss = model(
        token_ids=x_batch,
        targets=y_batch,
    )

    print("\nShape tracing:")
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())

    print("\nExpected logits shape:")
    print("(batch_size, block_size, vocab_size)")
    print((4, block_size, tokenizer.vocab_size))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_losses = []
    val_losses = []
    steps = []

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "builtin_tiny_gpt_best.pt",
    )

    best_val_loss = float("inf")

    print("\nTraining built-in tiny GPT-style model...\n")

    for step in range(1, max_steps + 1):
        x_batch, y_batch = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )

        logits, loss = model(
            token_ids=x_batch,
            targets=y_batch,
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step % eval_interval == 0 or step == 1:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
                eval_iters=eval_iters,
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            steps.append(step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )

            print(
                f"Step {step:5d}/{max_steps} | "
                f"train loss = {train_loss:.4f} | "
                f"val loss = {val_loss:.4f}"
            )

    print("\nBest validation loss:", best_val_loss)
    print("Saved checkpoint:", checkpoint_path)

    plt.figure()
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="validation loss")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Built-in Tiny GPT training curve")
    plt.legend()
    plt.show()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nFinal checkpoint:")
    print("Best step:", checkpoint["step"])
    print("Best val loss:", checkpoint["val_loss"])

    # Generate text
    prompt = "the "
    context_ids = tokenizer.encode(prompt)

    context = torch.tensor(
        [context_ids],
        dtype=torch.long,
        device=device,
    )

    generated = generate(
        model=model,
        context=context,
        max_new_tokens=300,
        temperature=0.8,
        top_k=5,
    )

    generated_text = tokenizer.decode(
        generated[0].cpu().tolist()
    )

    print("\nGenerated text:")
    print(generated_text)

    # Greedy-ish lower-temperature generation
    generated_low_temp = generate(
        model=model,
        context=context,
        max_new_tokens=300,
        temperature=0.5,
        top_k=3,
    )

    generated_low_temp_text = tokenizer.decode(
        generated_low_temp[0].cpu().tolist()
    )

    print("\nGenerated text with lower temperature:")
    print(generated_low_temp_text)


if __name__ == "__main__":
    main()