# Lesson 19: Causal Self-Attention and a Tiny GPT-style Language Model
#
# Goal:
#   Build a tiny decoder-only Transformer from scratch.
#
# Architecture:
#   token ids
#       -> token embeddings
#       -> positional embeddings
#       -> causal Transformer blocks
#       -> final LayerNorm
#       -> language-modeling head
#       -> logits over vocabulary
#
# Training objective:
#   next-token prediction
#
# If input is:
#   x = [t0, t1, t2, ..., t_{T-1}]
#
# target is:
#   y = [t1, t2, t3, ..., t_T]
#
# Causal self-attention ensures position i cannot attend to future positions.

import math
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

    This is intentionally tiny so training is fast.
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

    # Repeat so the model has enough examples.
    text = base_text.lower() * 200

    # Normalize whitespace.
    text = " ".join(text.split())

    return text


# ------------------------------------------------------------
# 3. Character tokenizer
# ------------------------------------------------------------

class CharacterTokenizer:
    """
    Minimal character-level tokenizer.

    stoi: string-to-index dictionary
    itos: index-to-string dictionary
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

    y is x shifted one position to the left in the original stream.
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

    x = x.to(device)
    y = y.to(device)

    return x, y


# ------------------------------------------------------------
# 6. Causal multi-head attention
# ------------------------------------------------------------

def scaled_dot_product_attention_multihead(Q, K, V, causal_mask):
    """
    Q, K, V:
        (B, H, T, head_dim)

    causal_mask:
        (1, 1, T, T)

    Returns:
        output:            (B, H, T, head_dim)
        attention_weights: (B, H, T, T)
    """

    B, H, T, head_dim = Q.shape

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    scores = scores.masked_fill(causal_mask == False, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.

    Input:
        x shape = (B, T, D)

    Output:
        output shape = (B, T, D)

    The causal mask prevents position i from attending to positions j > i.
    """

    def __init__(self, embedding_dim, num_heads, block_size, dropout_prob):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.block_size = block_size

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        self.attention_dropout = nn.Dropout(p=dropout_prob)
        self.residual_dropout = nn.Dropout(p=dropout_prob)

        # Register mask as buffer so it moves with model.to(device)
        # but is not a trainable parameter.
        causal_mask = torch.tril(
            torch.ones(block_size, block_size, dtype=torch.bool)
        )

        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, block_size, block_size),
        )

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

    def forward(self, x, return_attention=False):
        B, T, D = x.shape

        assert T <= self.block_size

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        causal_mask = self.causal_mask[:, :, :T, :T]

        head_outputs, attention_weights = scaled_dot_product_attention_multihead(
            Q=Q,
            K=K,
            V=V,
            causal_mask=causal_mask,
        )

        head_outputs = self.attention_dropout(head_outputs)

        concatenated = self.combine_heads(head_outputs)

        output = self.output_projection(concatenated)
        output = self.residual_dropout(output)

        if return_attention:
            return output, attention_weights

        return output, None


# ------------------------------------------------------------
# 7. Feedforward network
# ------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """
    Position-wise MLP used inside Transformer blocks.
    """

    def __init__(self, embedding_dim, feedforward_dim, dropout_prob):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(p=dropout_prob),
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# 8. GPT-style Transformer block
# ------------------------------------------------------------

class GPTBlock(nn.Module):
    """
    Decoder-only Transformer block with causal self-attention.

    Pre-norm architecture:

        x = x + causal_attention(LayerNorm(x))
        x = x + feedforward(LayerNorm(x))
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        feedforward_dim,
        block_size,
        dropout_prob,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = CausalMultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout_prob=dropout_prob,
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feedforward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            dropout_prob=dropout_prob,
        )

    def forward(self, x, return_attention=False):
        attention_output, attention_weights = self.attention(
            self.norm1(x),
            return_attention=return_attention,
        )

        x = x + attention_output

        feedforward_output = self.feedforward(self.norm2(x))
        x = x + feedforward_output

        return x, attention_weights


# ------------------------------------------------------------
# 9. Tiny GPT language model
# ------------------------------------------------------------

class TinyGPT(nn.Module):
    """
    Tiny decoder-only Transformer language model.

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

        self.blocks = nn.ModuleList([
            GPTBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                block_size=block_size,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, targets=None, return_attention=False):
        """
        token_ids:
            (B, T)

        targets:
            (B, T), optional

        If targets are provided, compute next-token loss.
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

        all_attention_weights = []

        for block in self.blocks:
            x, attention_weights = block(
                x,
                return_attention=return_attention,
            )

            if return_attention:
                all_attention_weights.append(attention_weights)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            # CrossEntropyLoss expects:
            #   input logits shape: (N, C)
            #   target shape:       (N,)
            #
            # Here:
            #   logits:  (B, T, vocab_size)
            #   targets: (B, T)
            #
            # So flatten B and T:
            #   logits_flat:  (B*T, vocab_size)
            #   targets_flat: (B*T,)
            logits_flat = logits.view(B * T, self.vocab_size)
            targets_flat = targets.view(B * T)

            loss = nn.functional.cross_entropy(
                logits_flat,
                targets_flat,
            )

        return logits, loss, all_attention_weights


# ------------------------------------------------------------
# 10. Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters):
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

            _, loss, _ = model(x_batch, targets=y_batch)
            losses.append(loss.item())

        results[split_name] = sum(losses) / len(losses)

    model.train()

    return results


# ------------------------------------------------------------
# 11. Text generation
# ------------------------------------------------------------

@torch.no_grad()
def generate(model, context, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressive sampling.

    context:
        token ids, shape (B, T)

    For each new token:
        1. crop context to block_size
        2. get logits
        3. take logits at final position
        4. sample next token
        5. append token to context
    """

    model.eval()

    for _ in range(max_new_tokens):
        context_cropped = context[:, -model.block_size:]

        logits, _, _ = model(context_cropped)

        # Use only the last position's logits.
        next_token_logits = logits[:, -1, :]

        next_token_logits = next_token_logits / temperature

        if top_k is not None:
            values, indices = torch.topk(next_token_logits, k=top_k)

            filtered_logits = torch.full_like(
                next_token_logits,
                fill_value=float("-inf"),
            )

            filtered_logits.scatter_(dim=1, index=indices, src=values)
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
# 12. Plot attention head
# ------------------------------------------------------------

def plot_attention_head(attention_weights, layer_index, head_index, title):
    """
    attention_weights:
        (B, H, T, T)
    """

    attention_np = attention_weights[0, head_index].detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(attention_np)
    plt.colorbar()
    plt.xlabel("Key/value position")
    plt.ylabel("Query position")
    plt.title(f"{title}\nLayer {layer_index}, Head {head_index}")
    plt.show()


# ------------------------------------------------------------
# 13. Count parameters
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
# 14. Main script
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

    model = TinyGPT(
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

    logits, loss, all_attention_weights = model(
        x_batch,
        targets=y_batch,
        return_attention=True,
    )

    print("\nShape tracing:")
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())
    print("number of attention tensors:", len(all_attention_weights))

    for layer_index, attention_weights in enumerate(all_attention_weights):
        print(
            f"Layer {layer_index} attention shape:",
            attention_weights.shape,
        )

    # Check causal attention matrix
    first_attention = all_attention_weights[0]
    print("\nFirst layer, head 0, first example attention matrix:")
    print(first_attention[0, 0])

    plot_attention_head(
        attention_weights=first_attention,
        layer_index=0,
        head_index=0,
        title="Causal attention before training",
    )

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
        "tiny_gpt_best.pt",
    )

    best_val_loss = float("inf")

    print("\nTraining tiny GPT-style model...\n")

    for step in range(1, max_steps + 1):
        x_batch, y_batch = get_batch(
            data=train_data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )

        logits, loss, _ = model(x_batch, targets=y_batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    plt.title("Tiny GPT training curve")
    plt.legend()
    plt.show()

    # Load best checkpoint
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

    generated_text = tokenizer.decode(generated[0].cpu().tolist())

    print("\nGenerated text:")
    print(generated_text)

    # Attention after training
    x_batch, y_batch = get_batch(
        data=val_data,
        batch_size=4,
        block_size=block_size,
        device=device,
    )

    logits, loss, all_attention_weights = model(
        x_batch,
        targets=y_batch,
        return_attention=True,
    )

    print("\nValidation batch loss after training:", loss.item())

    plot_attention_head(
        attention_weights=all_attention_weights[-1],
        layer_index=num_layers - 1,
        head_index=0,
        title="Causal attention after training",
    )


if __name__ == "__main__":
    main()