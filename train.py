from typing import Any, Tuple, Mapping
from absl import logging

import input_pipeline
import ml_collections
import optax
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from clu import metric_writers
from flax.training import train_state
from flax import jax_utils as flax_utils
import input_pipeline
from flax.training import common_utils
import numpy as np
import time
import functools
import model as transformer
import utils
import config


class GrokTrainState(train_state.TrainState):
  key: jax.random.KeyArray


def initial_state(rng: jax.random.PRNGKey,
                  config: ml_collections.ConfigDict,
                  learning_rate_fn,
                  vocab_size: int,
                  input: Mapping[str, jax.Array]) -> GrokTrainState:
  model = transformer.DecoderTransformer(
      vocab_size=vocab_size,
      n_embed=config.n_embed,
      n_heads=config.n_heads,
      max_block_size=config.max_block_size,
      dropout_rate=config.dropout_rate,
      num_decoder_blocks=config.num_decoder_blocks,)
  rng, model_rng = jax.random.split(rng)
  init_variables = model.init(model_rng, input['x'], training=False)

  tx = optax.adam(learning_rate=learning_rate_fn)

  state = GrokTrainState.create(
      apply_fn=model.apply, params=init_variables['params'], tx=tx, key=rng)
  return state


def train_step(dropout_key, state: train_state.TrainState,
               batch: Mapping[str, jax.Array],
               learning_rate_fn) -> Tuple[train_state.TrainState,
                                          dict[str, jax.Array]]:
  labels = batch['y']

  def loss_fn(params) -> tuple[jax.Array, Any]:
    logits = state.apply_fn({'params': params}, batch['x'], training=True,
                            rngs={'dropout': dropout_key}
                            )
    # apply loss only on the logits for the last token
    logits = logits[:, -1, :]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.mean(loss), logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(state.params)

  new_state = state.apply_gradients(grads=grad)
  metrics = utils.compute_metrics(logits, labels)
  metrics["learning_rate"] = learning_rate_fn(state.step)

  return new_state, metrics


@functools.partial(jax.pmap, axis_name="batch")
def eval_step(state, batch):
  logits = state.apply_fn({'params': state.params}, batch['x'], training=False)
  return utils.compute_metrics(logits[:, -1, :], batch['y'])


def evaluate(state: train_state.TrainState, batch: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
  batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch))
  metrics = eval_step(state, batch)
  metrics = flax_utils.unreplicate(metrics)
  return metrics


def make_learning_rate_fn(train_config):
  def learning_rate_fn(step: int):
    del step
    return train_config.learning_rate

  return learning_rate_fn


def train_and_evaluate(workdir: str, train_config) -> None:
  rng = jax.random.PRNGKey(train_config.init_seed)
  train_ds, eval_ds = input_pipeline.input_pipeline(
    train_config.equation_type, train_config.prime, train_config.train_fraction)

  rng, ds_rng = jax.random.split(rng)
  init_example = train_ds.get_batch(ds_rng, 1)
  rng, model_rng = jax.random.split(rng)

  learning_rate_fn = make_learning_rate_fn(train_config)
  state = initial_state(model_rng, train_config, learning_rate_fn,
                        train_ds.vocab_size, init_example)

  if train_config.restore_checkpoints:
    state = checkpoints.restore_checkpoint(workdir, state)
    logging.info(f"Restored checkpoint from step: {state.step}.")

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0, asynchronous=False)
  if int(state.step) == 0:
    writer.write_hparams(train_config.to_dict())

  # Replicate state.
  state = flax_utils.replicate(state)
  rng, dropout_key = jax.random.split(rng)

  p_train_step = jax.pmap(
      functools.partial(
          train_step, dropout_key=dropout_key,
          learning_rate_fn=learning_rate_fn),
      axis_name='batch')

  train_metrics_last_t = time.time()
  for step in range(state.step, train_config.train_steps):
    batch_rng = jax.random.fold_in(rng, step)
    batch = train_ds.get_batch(batch_rng, train_config.batch_size)
    batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch))

    state, metrics = p_train_step(state=state, batch=batch)
    if step % train_config.log_train_metrics == 0:
      eval_metrics = evaluate(state, eval_ds.get_batch(
        batch_rng, train_config.eval_batch_size))
      summary = {
          f'train_{k}': v
          for k, v in jax.tree_util.tree_map(lambda x: x.mean(), metrics).items()
      }
      summary.update({
          f'eval_{k}': v
          for k, v in jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics).items()
      })
      summary['steps_per_second'] = (train_config.log_train_metrics *
                                     train_config.batch_size) / (time.time() - train_metrics_last_t)

      logging.info(f'Step {step}: {summary}')
      writer.write_scalars(step, summary)
      train_metrics_last_t = time.time()

      if jax.process_index() == 0:
        checkpoints.save_checkpoint(workdir, flax_utils.unreplicate(state),
                                    train_config.train_steps, overwrite=True)


train_config = config.get_train_config()
train_and_evaluate(f"/tmp/grokking_training/grok_{train_config.equation_type}_prime_{train_config.prime}_v0",
                   train_config=train_config),
