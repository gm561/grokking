import jax
import jax.numpy as jnp

from flax import linen as nn


class MLP(nn.Module):
  n_embed: int
  n_out: int
  dropout: float = 0.1

  @nn.compact
  def __call__(self, x, training: bool):
    x = nn.Dense(features=self.n_embed)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.n_out)(x)
    x = nn.Dropout(rate=0.1, )(x, deterministic=not training)
    return x


class DecoderBlock(nn.Module):
  n_embed: int
  n_heads: int = 1
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, x, mask, training):
    x = nn.LayerNorm()(x)
    x = x + nn.SelfAttention(
        num_heads=self.n_heads, qkv_features=self.n_embed,
        deterministic=not training, dropout_rate=self.dropout_rate)(
        x, mask=mask)
    x = nn.LayerNorm()(x)
    x = x + MLP(n_embed=4 * self.n_embed, n_out=self.n_embed)(x, training)

    return x


class DecoderTransformer(nn.Module):
  vocab_size: int
  n_embed: int
  max_block_size: int
  dropout_rate: float
  num_decoder_blocks: int
  n_heads: int

  @nn.compact
  def __call__(self, x: jax.Array, training: bool) -> jax.Array:
    _, block_size = x.shape
    token_embed = nn.Embed(num_embeddings=self.vocab_size,
                           features=self.n_embed)(x)
    pos_embed = nn.Embed(
        num_embeddings=self.max_block_size, features=self.n_embed)(
        jnp.arange(0, block_size))
    mask = nn.make_causal_mask(x)

    x = token_embed + pos_embed

    for _ in range(self.num_decoder_blocks):
      x = DecoderBlock(
          n_embed=self.n_embed, dropout_rate=self.dropout_rate,
          n_heads=self.n_heads)(
          x, mask=mask, training=training)

    x = nn.LayerNorm()(x)

    logits = nn.Dense(features=self.vocab_size)(x)

    return logits
