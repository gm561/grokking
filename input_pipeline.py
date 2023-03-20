from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from operator import itemgetter
import random

Array = Any


def make_equation(x, y, z):
  return (str(x), 'o', str(y), '=', str(z))


def all_operations_in_p(operation: str, prime: int) -> tuple[str, str]:
  all_arithmetic_operations = {
      "x+y": lambda x, y, prime: make_equation(x, y, (x + y) % prime),
      "x-y": lambda x, y, prime: make_equation(x, y, (x - y) % prime),
      # to account for 0 division, we set y to 1 if 0.
      "x/y": lambda x, y, prime: make_equation(x, y, (x / max(y, 1)) % prime),
      "x^2+y^2": lambda x, y, prime: make_equation(x, y, (x**2 + y**2) % prime),
  }

  xs, ys = sorted(set(range(prime))), sorted(set(range(prime)))
  xs, ys = [i.reshape(-1) for i in np.meshgrid(xs, ys)]
  if operation not in all_arithmetic_operations:
    raise ValueError(f"Operation {operation} not supported")
  ops = [all_arithmetic_operations[operation](x, y, prime) for x, y in
         zip(xs, ys)]
  random.seed(42)
  random.shuffle(ops)
  return ops


class DataGenerator:
  def __init__(self, elements: Array) -> None:
    self._elements = elements
    self._sequence_length = 5
    self._vocab = sorted(set([i for e in elements for i in e]))
    self._vocab2idx = {c: i for i, c in enumerate(self._vocab)}
    self._idx2vocab = {i: c for c, i in self._vocab2idx.items()}

    self._encoded_elements = self.encode(elements)

  @ property
  def vocab_size(self) -> int:
    return len(self._vocab)

  @ property
  def sequence_length(self) -> int:
    return self._sequence_length

  def encode(self, inputs: Array) -> jax.Array:
    def encode_str(s: str) -> jax.Array:
      tokens = [self._vocab2idx[c] for c in s]
      tokens_length = len(tokens)
      if tokens_length != self._sequence_length:
        raise ValueError(
            f"Input length: is wrong : {tokens_length} expected: {self._sequence_length}")
      return tokens

    return [encode_str(single_input) for single_input in inputs]

  def decode(self, batch_inputs: Array) -> np.ndarray:
    def decode_inputs(inputs):
      decoded = []
      for elem in inputs.tolist():
        decoded.append(self._idx2vocab[elem])
      return decoded

    return list(map(decode_inputs, batch_inputs))

  def get_batch(self, rng, batch_size) -> tuple[jax.Array, jax.Array]:
    perm = jax.random.permutation(
        rng, max(batch_size, len(self._elements))).astype(
        jnp.int32)[:batch_size]
    perm = jax.numpy.mod(perm, len(self._elements))
    batch = jnp.asarray(itemgetter(
      *perm)(self._encoded_elements)).reshape(batch_size, -1)
    return {'x': batch[:, :-1], 'y': batch[:, -1:].reshape(-1)}


def input_pipeline(operation: str, prime: int, train_fraction: float) -> tuple[
        DataGenerator, DataGenerator]:
  elements = all_operations_in_p(operation, prime)
  train_set = elements[: int(len(elements) * train_fraction)]
  eval_set = elements[int(len(elements) * train_fraction):]

  return DataGenerator(train_set), DataGenerator(eval_set)
