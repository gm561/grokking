import optax
import jax
from jax import lax
import jax.numpy as jnp


def compute_metrics(logits: jax.Array, labels: jax.Array) -> dict[str, jax.Array]:
  # Computes sequence accuracy, which is the same as the accuracy during
  # inference, since teacher forcing is irrelevant when all output are correct.
  token_accuracy = jnp.argmax(logits, -1) == labels
  loss = jnp.mean(
      optax.softmax_cross_entropy_with_integer_labels(
          logits=logits, labels=labels))
  accuracy = jnp.mean(token_accuracy)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')

  return metrics
