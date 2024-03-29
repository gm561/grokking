from flax import struct

import ml_collections


def get_train_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()
  config.learning_rate = 0.001
  config.restore_checkpoints = True
  config.init_seed = 0
  config.train_steps = 10**6
  config.batch_size = 512
  config.eval_batch_size = 512
  config.log_train_metrics = 100

  config.n_heads = 4
  config.n_embed = 128
  config.max_block_size = 5
  config.dropout_rate = 0.1
  config.num_decoder_blocks = 2

  config.equation_type = 'x+y'
  config.train_fraction = 0.4
  config.prime = 97

  return config
