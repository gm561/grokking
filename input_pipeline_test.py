from absl.testing import parameterized

import input_pipeline
import jax
import chex
import jax.numpy as jnp


class InputPipelineTest(parameterized.TestCase):

  def test_base_case(self):
    train_set, eval_set = input_pipeline.input_pipeline(
        operation="x+y", prime=7, train_fraction=0.5)

    train_batch_size = 5
    test_batch_size = 1
    rng = jax.random.PRNGKey(0)
    train_example = train_set.get_batch(rng, batch_size=train_batch_size)
    test_example = eval_set.get_batch(rng, batch_size=test_batch_size)

    chex.assert_shape(train_example['x'], (train_batch_size, 4))
    chex.assert_shape(test_example['x'], (test_batch_size, 4))

    chex.assert_shape(train_example['y'], (train_batch_size,))
    chex.assert_shape(test_example['y'], (test_batch_size,))

    chex.assert_trees_all_equal(
        train_example['x'][: 2],
        jnp.asarray([[1, 8, 5, 7],
                     [4, 8, 3, 7]], dtype=jnp.int32))

    chex.assert_trees_all_equal(
        train_example['y'][: 2],
        jnp.asarray([6, 0], dtype=jnp.int32))

    chex.assert_trees_all_equal(
        train_set.decode(test_example['x']),
        train_set.decode(jnp.asarray([[4, 8, 5, 7]], dtype=jnp.int32)))

    chex.assert_trees_all_equal(
        train_set.decode([test_example['y']]),
        train_set.decode(jnp.asarray([[2]], dtype=jnp.int32)))
