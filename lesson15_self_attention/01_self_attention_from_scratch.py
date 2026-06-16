# Lesson 15: Self-Attention from Scratch
#
# Goal:
#   Implement scaled dot-product self-attention manually in PyTorch.
#
# We will:
#   1. Create a toy batch of token embeddings.
#   2. Project embeddings into queries, keys, and values.
#   3. Compute attention scores Q K^T / sqrt(d_k).
#   4. Apply softmax to get attention weights.
#   5. Compute weighted sums of values.
#   6. Verify shapes and row sums.
#
# This is the core mechanism behind Transformers.

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
# 2. Scaled dot-product attention
# ------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Parameters
    ----------
    Q : torch.Tensor
        Query tensor, shape (B, T, D)

    K : torch.Tensor
        Key tensor, shape (B, T, D)

    V : torch.Tensor
        Value tensor, shape (B, T, D)

    mask : torch.Tensor or None
        Optional mask, shape broadcastable to (B, T, T).
        Mask entries should be True where attention is allowed
        and False where attention should be blocked.

    Returns
    -------
    output : torch.Tensor
        Attention output, shape (B, T, D)

    attention_weights : torch.Tensor
        Attention weights, shape (B, T, T)
    """

    assert Q.ndim == 3
    assert K.ndim == 3
    assert V.ndim == 3
    assert Q.shape == K.shape == V.shape

    B, T, D = Q.shape

    # scores[b, i, j] = dot product between query of token i
    # and key of token j.
    scores = Q @ K.transpose(-2, -1)

    # Scale scores to avoid extremely large dot products.
    scores = scores / math.sqrt(D)

    if mask is not None:
        # Fill disallowed positions with a very negative number,
        # so softmax gives them probability approximately zero.
        scores = scores.masked_fill(mask == False, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)

    output = attention_weights @ V

    return output, attention_weights


# ------------------------------------------------------------
# 3. Self-attention module
# ------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Single-head self-attention layer.

    Input:
        x shape = (B, T, D)

    Output:
        output shape = (B, T, D)
        attention_weights shape = (B, T, T)

    This layer learns three linear maps:
        W_Q, W_K, W_V
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        assert x.ndim == 3

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        output, attention_weights = scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            mask=mask,
        )

        return output, attention_weights


# ------------------------------------------------------------
# 4. Create a causal mask
# ------------------------------------------------------------

def create_causal_mask(sequence_length, device):
    """
    Create lower-triangular causal mask.

    mask[i, j] = True  if token i is allowed to attend to token j
               = False if token i is NOT allowed to attend to token j

    In causal language modeling:
        token i may attend only to tokens j <= i.
    """

    mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    )

    mask = mask.to(device)

    return mask


# ------------------------------------------------------------
# 5. Plot attention matrix
# ------------------------------------------------------------

def plot_attention_matrix(attention_weights, title):
    """
    Plot attention matrix for the first example in the batch.

    attention_weights shape:
        (B, T, T)
    """

    attention_np = attention_weights[0].detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(attention_np)
    plt.colorbar()
    plt.xlabel("Key/value token index j")
    plt.ylabel("Query token index i")
    plt.title(title)
    plt.show()


# ------------------------------------------------------------
# 6. Main script
# ------------------------------------------------------------

def main():
    torch.manual_seed(0)

    device = get_device()
    print("Using device:", device)

    # Toy dimensions
    batch_size = 2
    sequence_length = 5
    embedding_dim = 8

    # Toy input:
    # batch of token embeddings
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
        device=device,
    )

    print("\nInput:")
    print("x shape:", x.shape)

    self_attention = SelfAttention(
        embedding_dim=embedding_dim,
    ).to(device)

    print("\nSelf-attention module:")
    print(self_attention)

    # --------------------------------------------------------
    # Non-causal self-attention
    # --------------------------------------------------------

    output, attention_weights = self_attention(x)

    print("\nNon-causal self-attention:")
    print("output shape:", output.shape)
    print("attention_weights shape:", attention_weights.shape)

    print("\nAttention row sums for first batch element:")
    print(torch.sum(attention_weights[0], dim=-1))

    print("\nFirst attention matrix:")
    print(attention_weights[0])

    plot_attention_matrix(
        attention_weights=attention_weights,
        title="Non-causal self-attention weights",
    )

    # --------------------------------------------------------
    # Causal self-attention
    # --------------------------------------------------------

    causal_mask = create_causal_mask(
        sequence_length=sequence_length,
        device=device,
    )

    print("\nCausal mask:")
    print(causal_mask)

    causal_output, causal_attention_weights = self_attention(
        x,
        mask=causal_mask,
    )

    print("\nCausal self-attention:")
    print("causal_output shape:", causal_output.shape)
    print("causal_attention_weights shape:", causal_attention_weights.shape)

    print("\nCausal attention row sums for first batch element:")
    print(torch.sum(causal_attention_weights[0], dim=-1))

    print("\nFirst causal attention matrix:")
    print(causal_attention_weights[0])

    plot_attention_matrix(
        attention_weights=causal_attention_weights,
        title="Causal self-attention weights",
    )


if __name__ == "__main__":
    main()