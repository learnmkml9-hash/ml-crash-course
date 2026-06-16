# Lesson 17: Transformer Encoder Block from Scratch
#
# Goal:
#   Build a Transformer encoder block using:
#       - multi-head self-attention
#       - residual connections
#       - layer normalization
#       - feedforward network
#       - dropout
#
# We use the pre-norm architecture:
#
#   x = x + dropout(MHA(LayerNorm(x)))
#   x = x + dropout(FFN(LayerNorm(x)))
#
# This block preserves shape:
#   input:  (B, T, D)
#   output: (B, T, D)

import math
import torch
import torch.nn as nn
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
# 2. Multi-head scaled dot-product attention
# ------------------------------------------------------------

def scaled_dot_product_attention_multihead(Q, K, V, mask=None):
    """
    Q, K, V shape:
        (B, H, T, head_dim)

    mask shape:
        broadcastable to (B, H, T, T)

    Returns:
        output shape:            (B, H, T, head_dim)
        attention_weights shape: (B, H, T, T)
    """

    assert Q.ndim == 4
    assert K.ndim == 4
    assert V.ndim == 4
    assert Q.shape == K.shape == V.shape

    B, H, T, head_dim = Q.shape

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights


# ------------------------------------------------------------
# 3. Multi-head self-attention module
# ------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Input:
        x shape = (B, T, D)

    Output:
        output shape = (B, T, D)
        attention_weights shape = (B, H, T, T)
    """

    def __init__(self, embedding_dim, num_heads, dropout_prob):
        super().__init__()

        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        self.attention_dropout = nn.Dropout(p=dropout_prob)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def split_heads(self, x):
        """
        Convert:
            (B, T, D)
        to:
            (B, H, T, head_dim)
        """

        B, T, D = x.shape
        assert D == self.embedding_dim

        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        """
        Convert:
            (B, H, T, head_dim)
        to:
            (B, T, D)
        """

        B, H, T, head_dim = x.shape

        assert H == self.num_heads
        assert head_dim == self.head_dim

        x = x.transpose(1, 2)
        x = x.contiguous().view(B, T, self.embedding_dim)

        return x

    def forward(self, x, mask=None):
        assert x.ndim == 3

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

        # Dropout on attention weights is common in Transformers.
        # To keep our manual attention function simple, we apply dropout to
        # the head outputs here instead.
        head_outputs = self.attention_dropout(head_outputs)

        concatenated = self.combine_heads(head_outputs)
        output = self.output_projection(concatenated)

        return output, attention_weights


# ------------------------------------------------------------
# 4. Position-wise feedforward network
# ------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """
    Position-wise feedforward network.

    Input:
        x shape = (B, T, D)

    Output:
        output shape = (B, T, D)

    It is called "position-wise" because the same MLP is applied independently
    to each token position.
    """

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
# 5. Transformer encoder block
# ------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block using pre-norm architecture.

    Input:
        x shape = (B, T, D)

    Output:
        x shape = (B, T, D)
        attention_weights shape = (B, H, T, T)
    """

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
        """
        Pre-norm Transformer encoder block:

            x = x + Dropout(MHA(LayerNorm(x)))
            x = x + Dropout(FFN(LayerNorm(x)))
        """

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
# 6. Stack of Transformer encoder blocks
# ------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Stack multiple Transformer encoder blocks.

    Input:
        x shape = (B, T, D)

    Output:
        x shape = (B, T, D)
        all_attention_weights: list of length num_layers
    """

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
# 7. Masks
# ------------------------------------------------------------

def create_padding_mask(token_ids, pad_token_id=0):
    """
    token_ids shape:
        (B, T)

    Returns:
        padding_mask shape = (B, 1, 1, T)

    True means valid token.
    False means padding token.

    This masks key/value positions.
    """

    mask = token_ids != pad_token_id
    mask = mask.unsqueeze(1).unsqueeze(2)

    return mask


# ------------------------------------------------------------
# 8. Plot attention head
# ------------------------------------------------------------

def plot_attention_head(attention_weights, layer_index, head_index, title):
    """
    attention_weights shape:
        (B, H, T, T)

    We plot the first batch element.
    """

    attention_np = attention_weights[0, head_index].detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(attention_np)
    plt.colorbar()
    plt.xlabel("Key/value token index j")
    plt.ylabel("Query token index i")
    plt.title(f"{title}\nLayer {layer_index}, Head {head_index}")
    plt.show()


# ------------------------------------------------------------
# 9. Print row sums
# ------------------------------------------------------------

def print_attention_row_sums(attention_weights, layer_index):
    """
    Verify that each attention row sums to 1.
    """

    B, H, T, _ = attention_weights.shape

    print(f"\nAttention row sums, layer {layer_index}, first batch element:")

    for head in range(H):
        row_sums = torch.sum(attention_weights[0, head], dim=-1)
        print(f"Head {head}:", row_sums)


# ------------------------------------------------------------
# 10. Count parameters
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
# 11. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Toy dimensions
    batch_size = 2
    sequence_length = 6
    embedding_dim = 32
    num_heads = 4
    feedforward_dim = 128
    num_layers = 2
    dropout_prob = 0.1

    assert embedding_dim % num_heads == 0

    # Toy token embeddings
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device,
    )

    print("\nInput:")
    print("x shape:", x.shape)

    # Example token ids with padding.
    # 0 denotes padding.
    token_ids = torch.tensor(
        [
            [5, 7, 9, 4, 3, 0],
            [2, 8, 1, 6, 0, 0],
        ],
        device=device,
    )

    padding_mask = create_padding_mask(
        token_ids=token_ids,
        pad_token_id=0,
    )

    print("\nPadding mask:")
    print("token_ids shape:", token_ids.shape)
    print("padding_mask shape:", padding_mask.shape)
    print("padding_mask first example:", padding_mask[0, 0, 0])

    # Single encoder block
    encoder_block = TransformerEncoderBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout_prob=dropout_prob,
    ).to(device)

    total_params, trainable_params = count_parameters(encoder_block)

    print("\nSingle Transformer encoder block:")
    print(encoder_block)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    encoder_block.train()
    block_output, block_attention_weights = encoder_block(
        x,
        mask=padding_mask,
    )

    print("\nSingle block output:")
    print("block_output shape:", block_output.shape)
    print("block_attention_weights shape:", block_attention_weights.shape)

    print_attention_row_sums(
        attention_weights=block_attention_weights,
        layer_index=0,
    )

    print("\nAttention weights for first example, head 0, query token 0:")
    print(block_attention_weights[0, 0, 0])

    plot_attention_head(
        attention_weights=block_attention_weights,
        layer_index=0,
        head_index=0,
        title="Single Transformer encoder block attention",
    )

    # Stack of encoder blocks
    encoder = TransformerEncoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
    ).to(device)

    total_params, trainable_params = count_parameters(encoder)

    print("\nTransformer encoder stack:")
    print(encoder)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    encoder.train()
    encoder_output, all_attention_weights = encoder(
        x,
        mask=padding_mask,
    )

    print("\nEncoder stack output:")
    print("encoder_output shape:", encoder_output.shape)
    print("number of attention tensors:", len(all_attention_weights))

    for layer_index, attention_weights in enumerate(all_attention_weights):
        print(
            f"Layer {layer_index} attention shape:",
            attention_weights.shape,
        )

    # Evaluation mode disables dropout.
    encoder.eval()

    with torch.no_grad():
        eval_output_1, _ = encoder(x, mask=padding_mask)
        eval_output_2, _ = encoder(x, mask=padding_mask)

    max_difference = torch.max(torch.abs(eval_output_1 - eval_output_2)).item()

    print("\nDropout check:")
    print("Max difference between two eval-mode outputs:", max_difference)
    print("This should be 0 or extremely close to 0.")

    # Training mode keeps dropout active.
    encoder.train()

    train_output_1, _ = encoder(x, mask=padding_mask)
    train_output_2, _ = encoder(x, mask=padding_mask)

    train_max_difference = torch.max(
        torch.abs(train_output_1 - train_output_2)
    ).item()

    print("\nTraining-mode dropout check:")
    print("Max difference between two train-mode outputs:", train_max_difference)
    print("This should usually be nonzero because dropout is active.")


if __name__ == "__main__":
    main()