# Lesson 16: Multi-Head Attention from Scratch
#
# Goal:
#   Implement multi-head self-attention manually in PyTorch.
#
# We will:
#   1. Start with token embeddings x of shape (B, T, D).
#   2. Project x into Q, K, V.
#   3. Split Q, K, V into multiple heads.
#   4. Apply scaled dot-product attention independently in each head.
#   5. Concatenate heads.
#   6. Apply an output projection.
#   7. Compare non-causal and causal multi-head attention.

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
# 2. Scaled dot-product attention for multiple heads
# ------------------------------------------------------------

def scaled_dot_product_attention_multihead(Q, K, V, mask=None):
    """
    Multi-head scaled dot-product attention.

    Parameters
    ----------
    Q : torch.Tensor
        Query tensor, shape (B, H, T, head_dim)

    K : torch.Tensor
        Key tensor, shape (B, H, T, head_dim)

    V : torch.Tensor
        Value tensor, shape (B, H, T, head_dim)

    mask : torch.Tensor or None
        Optional mask. Shape should be broadcastable to (B, H, T, T).
        True means attention is allowed.
        False means attention is blocked.

    Returns
    -------
    output : torch.Tensor
        Output tensor, shape (B, H, T, head_dim)

    attention_weights : torch.Tensor
        Attention weights, shape (B, H, T, T)
    """

    assert Q.ndim == 4
    assert K.ndim == 4
    assert V.ndim == 4
    assert Q.shape == K.shape == V.shape

    B, H, T, head_dim = Q.shape

    # scores[b, h, i, j] =
    #   dot product between query token i and key token j
    #   inside head h.
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
    Multi-head self-attention from scratch.

    Input:
        x shape = (B, T, D)

    Output:
        output shape = (B, T, D)
        attention_weights shape = (B, H, T, T)
    """

    def __init__(self, embedding_dim, num_heads):
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

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def split_heads(self, x):
        """
        Split embedding dimension into multiple heads.

        Input:
            x shape = (B, T, D)

        Output:
            x shape = (B, H, T, head_dim)
        """

        B, T, D = x.shape

        assert D == self.embedding_dim

        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        """
        Concatenate multiple heads back into one embedding dimension.

        Input:
            x shape = (B, H, T, head_dim)

        Output:
            x shape = (B, T, D)
        """

        B, H, T, head_dim = x.shape

        assert H == self.num_heads
        assert head_dim == self.head_dim

        x = x.transpose(1, 2)

        # After transpose, memory may be non-contiguous.
        # contiguous() makes it safe to view.
        x = x.contiguous().view(B, T, self.embedding_dim)

        return x

    def forward(self, x, mask=None):
        """
        x shape:
            (B, T, D)

        mask:
            optional, broadcastable to (B, H, T, T)
        """

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

        concatenated = self.combine_heads(head_outputs)

        output = self.output_projection(concatenated)

        return output, attention_weights


# ------------------------------------------------------------
# 4. Causal mask
# ------------------------------------------------------------

def create_causal_mask(sequence_length, device):
    """
    Create causal mask for autoregressive attention.

    Initial shape:
        (T, T)

    Returned shape:
        (1, 1, T, T)

    This is broadcastable to:
        (B, H, T, T)
    """

    mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    )

    mask = mask.to(device)

    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


# ------------------------------------------------------------
# 5. Padding mask
# ------------------------------------------------------------

def create_padding_mask(token_ids, pad_token_id=0):
    """
    Create padding mask from token ids.

    token_ids shape:
        (B, T)

    Suppose pad_token_id = 0.
    True means valid token.
    False means padding token.

    Returned shape:
        (B, 1, 1, T)

    This is broadcastable to attention scores:
        (B, H, T, T)

    It masks key/value positions, not query positions.
    """

    mask = token_ids != pad_token_id
    mask = mask.unsqueeze(1).unsqueeze(2)

    return mask


# ------------------------------------------------------------
# 6. Plot attention for one head
# ------------------------------------------------------------

def plot_attention_head(attention_weights, head_index, title):
    """
    Plot attention matrix for the first batch element and one head.

    attention_weights shape:
        (B, H, T, T)
    """

    attention_np = attention_weights[0, head_index].detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(attention_np)
    plt.colorbar()
    plt.xlabel("Key/value token index j")
    plt.ylabel("Query token index i")
    plt.title(title)
    plt.show()


# ------------------------------------------------------------
# 7. Print attention row sums
# ------------------------------------------------------------

def print_attention_row_sums(attention_weights, name):
    """
    For every head, print row sums for first batch element.

    Each row should sum to approximately 1.
    """

    print(f"\n{name}: attention row sums for first batch element")

    B, H, T, _ = attention_weights.shape

    for head in range(H):
        row_sums = torch.sum(attention_weights[0, head], dim=-1)
        print(f"Head {head}:", row_sums)


# ------------------------------------------------------------
# 8. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Toy dimensions
    batch_size = 2
    sequence_length = 6
    embedding_dim = 16
    num_heads = 4

    assert embedding_dim % num_heads == 0

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device,
    )

    print("\nInput:")
    print("x shape:", x.shape)
    print("embedding_dim:", embedding_dim)
    print("num_heads:", num_heads)
    print("head_dim:", embedding_dim // num_heads)

    multihead_attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
    ).to(device)

    print("\nMulti-head attention module:")
    print(multihead_attention)

    # --------------------------------------------------------
    # Shape tracing manually
    # --------------------------------------------------------

    with torch.no_grad():
        Q = multihead_attention.query_projection(x)
        print("\nShape tracing:")
        print("Q before split:", Q.shape)

        Q_split = multihead_attention.split_heads(Q)
        print("Q after split:", Q_split.shape)

        Q_combined = multihead_attention.combine_heads(Q_split)
        print("Q after recombining:", Q_combined.shape)

    # --------------------------------------------------------
    # Non-causal multi-head self-attention
    # --------------------------------------------------------

    output, attention_weights = multihead_attention(x)

    print("\nNon-causal multi-head attention:")
    print("output shape:", output.shape)
    print("attention_weights shape:", attention_weights.shape)

    print_attention_row_sums(
        attention_weights=attention_weights,
        name="Non-causal",
    )

    plot_attention_head(
        attention_weights=attention_weights,
        head_index=0,
        title="Non-causal attention: head 0",
    )

    plot_attention_head(
        attention_weights=attention_weights,
        head_index=1,
        title="Non-causal attention: head 1",
    )

    # --------------------------------------------------------
    # Causal multi-head self-attention
    # --------------------------------------------------------

    causal_mask = create_causal_mask(
        sequence_length=sequence_length,
        device=device,
    )

    print("\nCausal mask shape:", causal_mask.shape)
    print("Causal mask:")
    print(causal_mask[0, 0])

    causal_output, causal_attention_weights = multihead_attention(
        x,
        mask=causal_mask,
    )

    print("\nCausal multi-head attention:")
    print("causal_output shape:", causal_output.shape)
    print("causal_attention_weights shape:", causal_attention_weights.shape)

    print_attention_row_sums(
        attention_weights=causal_attention_weights,
        name="Causal",
    )

    plot_attention_head(
        attention_weights=causal_attention_weights,
        head_index=0,
        title="Causal attention: head 0",
    )

    # --------------------------------------------------------
    # Padding mask example
    # --------------------------------------------------------

    token_ids = torch.tensor(
        [
            [5, 7, 9, 4, 0, 0],
            [3, 8, 1, 2, 6, 0],
        ],
        device=device,
    )

    padding_mask = create_padding_mask(
        token_ids=token_ids,
        pad_token_id=0,
    )

    print("\nPadding mask example:")
    print("token_ids shape:", token_ids.shape)
    print("padding_mask shape:", padding_mask.shape)
    print("padding_mask for first example:")
    print(padding_mask[0, 0, 0])

    padded_output, padded_attention_weights = multihead_attention(
        x,
        mask=padding_mask,
    )

    print("\nPadding-masked multi-head attention:")
    print("padded_output shape:", padded_output.shape)
    print("padded_attention_weights shape:", padded_attention_weights.shape)

    print("\nAttention weights for first example, head 0, query token 0:")
    print(padded_attention_weights[0, 0, 0])

    plot_attention_head(
        attention_weights=padded_attention_weights,
        head_index=0,
        title="Padding-masked attention: head 0",
    )


if __name__ == "__main__":
    main() 