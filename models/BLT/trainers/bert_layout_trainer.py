# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT Layout Trainer."""

#  pylint: disable=g-import-not-at-top
import functools
import sys
sys.path.append("..")

import time
from typing import Any, Dict, Optional

from absl import logging
from clu import metrics
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
from flax.training import common_utils
import jax
import jax.numpy as jnp
import input_pipeline
from nets import na_layout_net
from trainers import base_trainer
from utils import layout_bert_fast_decode
from utils import task_manager
import numpy as np
import os
from dataset import datasets_info


@flax.struct.dataclass
class TrainState:
  """Data structure for checkpoint the model."""
  step: int
  optimizer: flax.optim.Optimizer
  model_state: Optional[Any]


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """Metrics during training process."""
  loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  """Metrics during evaluation process."""
  eval_loss: metrics.Average.from_output("eval_loss")


def rmlm_masking(inputs, mask_token, pad_token):
  """Random length masking function.

  Different from standard BERT masking which has a fixed mask ratio. We follow
  masking process in mask predict (https://arxiv.org/abs/1904.09324). A random
  mask ratio between [0, 1) are sampled first and input sequence tokens will
  be masked based this mask ratio.

  Args:
    inputs: input layout sequences.
    mask_token: the index of mask token.
    pad_token: the index of pad token.
  Returns:
    dictionary of masked input, original input and mask weights.
  """
  targets = inputs

  rng = jax.random.PRNGKey(jnp.sum(inputs, dtype="int32"))

  # Gets positions to leave untouched.
  is_pad = inputs == pad_token
  lens = jnp.sum(~is_pad, axis=-1)
  # Random samples a mask ratio.
  mask_rate = 1 - jax.random.uniform(rng, lens.shape)
  # Obtains the ceiling of the lens to make sure we can mask at least one token.
  mask_lens = jax.lax.ceil(lens * mask_rate)
  # Positions to mask.
  rng, subrng = jax.random.split(rng)
  # Randomly generates the mask score uniformly.
  should_mask = jax.random.uniform(subrng, shape=inputs.shape)
  # Doesn't mask out padding.
  should_mask = jnp.where(is_pad, 2., should_mask)
  # should_mask = jnp.where(is_pad | (~target_mask), 2., should_mask)

  sorted_should_mask = jnp.sort(should_mask, axis=-1)

  # Obtains the cutoff score for the mask lens.
  cut_off = jnp.take_along_axis(
      sorted_should_mask, jnp.expand_dims(mask_lens-1, 1), axis=-1)
  cut_off = jnp.repeat(cut_off, inputs.shape[1], axis=1)

  # Scores smaller than the cutoff will be masked.
  should_mask = jnp.where(should_mask <= cut_off, 1., 0.)

  # Full array of MASK tokens
  fullmask = jnp.full_like(inputs, mask_token)

  # Only replace positions where `should_mask`
  masked_inputs = jnp.where(should_mask, fullmask, inputs)
  weights = should_mask
  return dict(masked_inputs=masked_inputs, targets=targets, weights=weights)


def attribute_random_masking(inputs, mask_token, pad_token, layout_dim):
  """Attribute-wise masking process..

  Different from standard BERT masking which has a fixed mask ratio. Each time,
  we only mask one of three attributes (category, size and position), then a
  random mask ratio between [0, 1) are sampled and this attirbute position
  tokens will be masked based this mask ratio.

  Args:
    inputs: input layout sequences.
    mask_token: the index of mask token.
    pad_token: the index of pad token.
    layout_dim: the dimension of layout.
  Returns:
    dictionary of masked input, original input and mask weights.
  """
  targets = inputs
  total_dim = layout_dim * 2 + 1

  rng = jax.random.PRNGKey(jnp.sum(inputs, dtype="int32"))

  # Gets positions to leave untouched.
  is_pad = inputs == pad_token
  position_ids = jnp.arange(inputs.shape[-1])[None, :]
  is_asset = position_ids % total_dim == 0
  # is_size = (position_ids % 5 == 1) | (position_ids % 5 == 2)
  # is_position = (position_ids % 5 == 3) | (position_ids % 5 == 4)
  is_size = functools.reduce(
      lambda x, y: x | y,
      [position_ids % total_dim == i for i in range(1, layout_dim + 1)])
  is_position = functools.reduce(
      lambda x, y: x | y,
      [position_ids % total_dim == i for i in range(layout_dim + 1, total_dim)])
  # three steps masking
  rand = jax.random.uniform(rng, (inputs.shape[0], 1))

  target_mask = (~is_pad) & is_asset
  target_mask = jnp.where(
      jnp.logical_and(rand >= 0.2, rand < 0.4),
      (is_asset | is_size) & (~is_pad), target_mask)
  # target_mask = jnp.where(
  #     jnp.logical_and(rand >= 0.4, rand < 0.6),
  #     (is_asset | is_position) & (~is_pad), target_mask)
  target_mask = jnp.where(rand >= 0.4, ~is_pad, target_mask)
  should_mask = target_mask

  # Full array of MASK tokens
  fullmask = jnp.full_like(inputs, mask_token)
  fullmask = jnp.where(is_pad, pad_token, fullmask)

  # Only replace positions where `should_mask`
  pre_masked_inputs = jnp.where(should_mask, inputs, fullmask)
  weights = is_asset & (~is_pad)
  weights = jnp.where(
      jnp.logical_and(rand >= 0.2, rand < 0.4), is_size & (~is_pad), weights)
  weights = jnp.where(
      jnp.logical_and(rand >= 0.4, rand < 0.6), is_position & (~is_pad),
      weights)
  weights = jnp.where(
      jnp.logical_and(rand >= 0.6, rand < 0.8), is_size & (~is_pad), weights)
  weights = jnp.where(rand >= 0.8, is_asset & (~is_pad), weights)

  # lens = jnp.sum(target_mask & (~is_pad), axis=-1)
  lens = jnp.sum(weights, axis=-1)
  rng, subrng = jax.random.split(rng)
  mask_rate = 1 - jax.random.uniform(subrng, lens.shape)

  # Obtains the ceiling of the lens to make sure we can mask at least one token.
  mask_lens = jax.lax.ceil(lens * mask_rate)
  # Positions to mask.
  rng, subrng = jax.random.split(rng)
  # Randomly generates the mask score uniformly.
  should_mask = jax.random.uniform(subrng, shape=inputs.shape)
  # Doesn't mask out padding.
  should_mask = jnp.where(weights, should_mask, 2.)

  sorted_should_mask = jnp.sort(should_mask, axis=-1)

  # Obtains the cutoff score for the mask lens.
  cut_off = jnp.take_along_axis(
      sorted_should_mask, jnp.expand_dims(mask_lens-1, 1), axis=-1)
  cut_off = jnp.repeat(cut_off, inputs.shape[1], axis=1)

  # Scores smaller than the cutoff will be masked.
  should_mask = jnp.where(should_mask <= cut_off, 1., 0.)

  # Full array of MASK tokens
  fullmask = jnp.full_like(inputs, mask_token)

  # Only replace positions where `should_mask`
  masked_inputs = jnp.where(should_mask, fullmask, pre_masked_inputs)
  weights = jnp.where(is_pad, 0, should_mask)
  return dict(masked_inputs=masked_inputs, targets=targets, weights=weights)


class BERTLayoutTrainer(base_trainer.LayoutBaseTrainer):
  """BERT-style Layout Trainer."""

  def preprocess_batch(self, batch, batch_size, dataset, use_vertical=False):
    label = None
    # When we reach to the end of the dataset iter, the batch size may not be
    # be our expected one. In the case, we simply skip it.
    if batch.shape[0] != batch_size:
      return None, None
    batch = attribute_random_masking(
        batch, mask_token=3, pad_token=0, layout_dim=self.layout_dim)
    batch = common_utils.shard(batch)
    return batch, label

  def create_train_state(
      self,
      rng,
      inputs,
  ):
    model = functools.partial(
        na_layout_net.NALayoutNet,
        use_vertical=self.config.use_vertical_info,
        vocab_size=self.config.vocab_size,
        hidden_size=self.config.qkv_dim,
        num_hidden_layers=self.config.num_layers,
        num_attention_heads=self.config.num_heads,
        intermediate_size=self.config.mlp_dim,
        pad_token_id=0,
        layout_dim=self.layout_dim)
    param_rng, dropout_rng, rng = jax.random.split(rng, 3)
    model_variables = model().init({
        "params": param_rng,
        "dropout": dropout_rng
    },
                                   inputs["inputs"],
                                   inputs["labels"],
                                   deterministic=False)
    model_state = dict(model_variables)
    model_params = model_state.pop("params")
    logging.info("logging model parameters")
    parameter_overview.log_parameter_overview(model_params)
    optimizer = self.create_optimizer().create(model_params)
    model_dict = dict(model=model)
    train_state = TrainState(
        step=0,
        optimizer=optimizer,
        model_state=model_state)
    return model_dict, train_state

  def train_step(self, rng, state, batch, label, learning_rate_fn, model_dict,
                 logits_mask):
    """Perform a single training step.

    Args:
      rng: The random seed,
      state: State of the model (optimizer and state).
      batch: Training inputs for this step.
      label: Training input vectical info (always None for now).
      learning_rate_fn: The learning scheduler.
      model_dict: The model used in training.
      logits_mask: Logits mask for each step.

    Returns:
      The new model state and dictionary with metrics
    """
    logging.info("train_step(batch=%s)", batch)
    step = state.step + 1
    lr = learning_rate_fn(state.step)
    model = model_dict["model"]

    def loss_fn(params):
      variables = {"params": params}
      variables.update(state.model_state)

      logits, new_variables = model().apply(
          {"params": params},
          batch["masked_inputs"],
          labels=label,
          deterministic=False,
          rngs={"dropout": rng},
          mutable=True)
      ce_loss, num_tokens = self.compute_weighted_cross_entropy(
          logits, batch["targets"], batch["weights"],
          self.config.label_smoothing, logits_mask)

      loss = ce_loss / num_tokens
      new_model_state = dict(new_variables)
      new_model_state.pop("params")
      return loss, new_model_state

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_model_state), grad = grad_fn(state.optimizer.target)
    grad = jax.lax.pmean(grad, "batch")
    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_state = state.replace(
        step=step, optimizer=new_optimizer, model_state=new_model_state)
    metrics_update = TrainMetrics.gather_from_model_output(loss=loss)
    return new_state, metrics_update

  def eval_step(self, rng, state, batch, label, model_dict, logits_mask):
    model = model_dict["model"]
    logits = model().apply(
        {"params": state.optimizer.target},
        batch["masked_inputs"],
        labels=label,
        deterministic=True)
    ce_loss, num_tokens = self.compute_weighted_cross_entropy(
        logits, batch["targets"], batch["weights"], self.config.label_smoothing,
        logits_mask)

    loss = ce_loss / num_tokens
    metrics_update = EvalMetrics.gather_from_model_output(eval_loss=loss)
    return metrics_update

  def test(self,
           batch_size=1,
           iterative_nums=None,
           conditional="none",
           max_decode_len=128,
           use_vertical=False,
           sample_step_num=100,
           prior=None,
           max_asset_num=22,
           vertical_idx=0):
    """Runs a test run."""
    rng = jax.random.PRNGKey(self.config.seed)
    np.random.seed(self.config.seed)
    # Make sure each host uses a different RNG.
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, model_rng, data_rng = jax.random.split(rng, 3)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    dataset = self.config.dataset
    test_ds, vocab_size, pos_info = input_pipeline.get_dataset(
        batch_size,
        self.config.dataset_path,
        jax.local_device_count(),
        "test/json_data",
        max_decode_len,
        add_bos=False,
        shuffle=self.config.train_shuffle,
        dataset_name=dataset,
        composition=self.config.composition,
        sort_by=self.config.sort_by)
    logits_mask, offset = self.make_mask(vocab_size, pos_info,
                                         max_decode_len,
                                         self.layout_dim)
    init_batch = jnp.ones(
        (batch_size, max_decode_len))
    init_label = jnp.ones((batch_size, 1))
    init_batch = dict(inputs=init_batch, labels=init_label)
    model_dict, state = self.create_train_state(model_rng, init_batch)
    test_checkpoint_dir = os.path.join(self.workdir, "checkpoints")

    ckpt_path = test_checkpoint_dir
    state = task_manager.restore_checkpoint(state, ckpt_path)
    state = flax_utils.replicate(state)

    sample_one_batch_fn = functools.partial(
        self.sample_one_batch,
        pos_info=pos_info,
        iterative_num=iterative_nums,
        conditional=conditional,
        logits_mask=logits_mask)
    p_generate_batch = jax.pmap(
        functools.partial(
            sample_one_batch_fn,
            model_dict=model_dict,
        ),
        axis_name="batch")
    test_iter = iter(test_ds)  # pytype: disable=wrong-arg-types
    generated_sample_list, real_sample_list = [], []
    assert iterative_nums is not None and len(iterative_nums) == 3
    iterative_nums = np.array(iterative_nums)
    def tohost(x):
      """Collect batches from all devices to host and flatten batch dimensions."""
      n_device, n_batch, *remaining_dims = x.shape
      return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))

    if conditional == "none":
      total_time = 0.
      for idx in range(sample_step_num):
        if use_vertical:
          test_label = jnp.full((batch_size, 1), vertical_idx)
        else:
          test_label = None
        asset_num = np.random.choice(max_asset_num, batch_size, p=prior) + 1
        rng, sample_rng = jax.random.split(rng, 2)
        p_rng = jax.random.split(sample_rng, jax.local_device_count())

        # All mask symbols.
        asset_num = jnp.array(asset_num, dtype="int32")[Ellipsis, None]
        # element_num = asset_num * 5
        element_num = asset_num * self.total_dim
        masked_batch = jnp.full((batch_size, 128), 3)

        position_ids = jnp.arange(masked_batch.shape[-1])[None, :]
        # Pads masked batch.
        masked_batch = jnp.where(position_ids >= element_num, 0, masked_batch)
        masked_batch = common_utils.shard(masked_batch)
        test_label = common_utils.shard(test_label)

        p_rng = jax.random.split(rng, jax.local_device_count())
        start_time = time.time()

        sample_layouts = p_generate_batch(masked_batch, test_label, p_rng,
                                          state)
        total_time += time.time() - start_time
        start_time = time.time()
        sample_layouts = tohost(sample_layouts)
        generated_sample_list.append(sample_layouts - offset)

      generated_samples = jnp.concatenate(generated_sample_list, axis=0)
      real_samples = None
      logging.info("decoding time: (%.4f)", total_time)
      return generated_samples, real_samples

    for idx, test_batch in enumerate(test_iter):
      if idx >= sample_step_num:
        break
      asset_num = np.random.choice(max_asset_num, batch_size, p=prior) + 1
      rng, sample_rng = jax.random.split(rng, 2)
      p_rng = jax.random.split(sample_rng, jax.local_device_count())
      test_batch = jax.tree_map(lambda x: x._numpy(), test_batch)  # pylint: disable=protected-access
      test_batch, _ = self.preprocess_batch(test_batch, batch_size, dataset,
                                            use_vertical)
      if test_batch is None or (conditional == "none" and
                                idx == sample_step_num):
        break
      test_batch = tohost(test_batch["targets"])
      if use_vertical:
        test_label = jnp.full((batch_size, 1), vertical_idx)
      else:
        test_label = None

      # All mask symbols.
      if conditional != "none":
        # asset_num = jnp.sum(test_batch > 0, axis=1, keepdims=True) // 5
        asset_num = jnp.sum(
            test_batch > 0, axis=1, keepdims=True) // self.total_dim
      else:
        asset_num = jnp.array(asset_num, dtype="int32")[Ellipsis, None]
      # element_num = asset_num * 5
      element_num = asset_num * self.total_dim
      masked_batch = jnp.full_like(test_batch, 3)

      position_ids = jnp.arange(masked_batch.shape[-1])[None, :]
      # is_asset = position_ids % 5 == 0
      # is_size = (position_ids % 5 == 1) | (position_ids % 5 == 2)
      is_asset = position_ids % self.total_dim == 0
      is_size = functools.reduce(lambda x, y: x | y, [
          position_ids % self.total_dim == i
          for i in range(1, self.layout_dim + 1)
      ])
      if conditional == "a+s":
        masked_batch = jnp.where(is_asset | is_size, test_batch, masked_batch)
      elif conditional == "a":
        masked_batch = jnp.where(is_asset, test_batch, masked_batch)
      # Pads masked batch.
      masked_batch = jnp.where(position_ids >= element_num, 0, masked_batch)
      masked_batch = common_utils.shard(masked_batch)

      p_rng = jax.random.split(rng, jax.local_device_count())
      test_label = common_utils.shard(test_label)
      sample_layouts = p_generate_batch(masked_batch, test_label, p_rng, state)
      sample_layouts = tohost(sample_layouts)
      generated_sample_list.append(sample_layouts - offset)
      real_sample_list.append(test_batch - offset)
    generated_samples = jnp.concatenate(generated_sample_list, axis=0)
    real_samples = jnp.concatenate(real_sample_list, axis=0)
    return generated_samples, real_samples

  '''inference 결과를 배경 이미지와 함께 반환하는 함수
    Args:
      batch_size: 배치 사이즈
      iterative_nums: 반복 횟수
      conditional: decoding 방식
      max_decode_len: 최대 sequence 길이
      use_vertical: vertical 정보 사용 여부
      sample_step_num: test할 데이터 개수 (사용하지 않음)
      max_asset_num: 최대 asset 개수 (생성할 최대 bounding box 개수와 동일)
      idx: test dataset list에서 inference에 사용할 데이터의 index
    return:
      generated_samples: 생성된 sample
  '''
  def test_with_backgroundImage(self,
           batch_size=1,
           iterative_nums=None,
           conditional="none",
           max_decode_len=128,
           use_vertical=False,
           sample_step_num=10,
           prior=None,
           max_asset_num=22,
           vertical_idx=0,
           idx=None):
    """Runs a test with background image run."""
    rng = jax.random.PRNGKey(self.config.seed)
    np.random.seed(self.config.seed)
    # Make sure each host uses a different RNG.
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, model_rng, data_rng = jax.random.split(rng, 3)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    dataset = self.config.dataset

    init_batch = jnp.ones(
        (batch_size, max_decode_len))
    init_label = jnp.ones((batch_size, 1))
    init_batch = dict(inputs=init_batch, labels=init_label)
    model_dict, state = self.create_train_state(model_rng, init_batch)

    test_checkpoint_dir = os.path.join(self.workdir, "checkpoints")

    ckpt_path = test_checkpoint_dir
    state = task_manager.restore_checkpoint(state, ckpt_path)
    state = flax_utils.replicate(state)

    test_ds, vocab_size, pos_info, image_link = input_pipeline.get_dataset(
        batch_size,
        self.config.dataset_path,
        jax.local_device_count(),
        "test/json_data",
        max_decode_len,
        add_bos=False,
        shuffle=self.config.train_shuffle,
        dataset_name=dataset,
        idx=idx,
        is_background_test=True,
        composition=self.config.composition,
        sort_by=self.config.sort_by)
    
    logits_mask, offset = self.make_mask(vocab_size, pos_info,
                                         max_decode_len,
                                         self.layout_dim)

    sample_one_batch_fn = functools.partial(
        self.sample_one_batch,
        pos_info=pos_info,
        iterative_num=iterative_nums,
        conditional=conditional,
        logits_mask=logits_mask)
    p_generate_batch = jax.pmap(
        functools.partial(
            sample_one_batch_fn,
            model_dict=model_dict,
        ),
        axis_name="batch")
    
    test_iter = iter(test_ds)  # pytype: disable=wrong-arg-types
    generated_sample_list, real_sample_list = [], []
    assert iterative_nums is not None and len(iterative_nums) == 3
    iterative_nums = np.array(iterative_nums)
    def tohost(x):
      """Collect batches from all devices to host and flatten batch dimensions."""
      n_device, n_batch, *remaining_dims = x.shape
      return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))

    # sequence 배열에서 masking을 하지 않는 category id에 해당하는 부분의 index을 찾는 함수
    # return: index list
    def find_indices_of_no_mask_id(test_batch, values, is_asset):
      true_indices = [index for index, value in enumerate((test_batch[0]-offset[0]).tolist()) if value in values and is_asset[0][index]]
      return true_indices

    # masking을 하지 않는 category id가 위치하는 index 값 list를 바탕으로 masking하지 않는 부분을 추가하는 함수
    # seq: <category_id> <width> <height> <center x> <center y> 가 반복됨
    # 특정 category를 masking하지 않으면 이에 해당하는 bbox 정보도 masking하지 않음 
    # return: index list
    def add_no_mask_to_size_and_pos(true_indices):
      indices = []
      for idx in true_indices :
        indices += [idx, idx+1, idx+2, idx+3, idx+4]

      return indices
    
    # masking하지 않는 부분을 나타내는 배열
    # return: no mask-True, mask-False로 이루어진 배열 입력 sequence의 차원과 동일 
    def masking_fn(no_mask_indices, position_ids) :
      is_no_mask = functools.reduce(lambda x, y: x | y, [
          position_ids == idx
          for idx in no_mask_indices
      ])

      return is_no_mask

    # masking할 부분과 하지 않을 부분을 정하는 함수 
    # return: 특정 부분이 3으로 masking된 test_batch
    def custom_decode(img_true_indices, txt_true_indices, masked_batch, test_batch, is_asset, position_ids) :
      if len(img_true_indices) != 0 and len(txt_true_indices) != 0:             # text에 해당하는 id와 image에 해당하는 id가 존재할 때 
        mask_what = input("what do you want not to mask? (im or txt)")
        if mask_what == "im" :                                                  # image에 해당하는 부분을 masking하지 않음 
          no_mask_indices = add_no_mask_to_size_and_pos(img_true_indices)
          is_image = masking_fn(no_mask_indices, position_ids)
          res_batch = jnp.where(is_asset | is_image, test_batch, masked_batch)
        else :                                                                  # text에 해당하는 부분을 masking하지 않음 
          no_mask_indices = add_no_mask_to_size_and_pos(txt_true_indices)
          is_text = masking_fn(no_mask_indices, position_ids)
          res_batch = jnp.where(is_asset | is_text, test_batch, masked_batch)
      elif len(txt_true_indices) != 0 :                                         # text만 있는 경우는 입력 seq의 첫 번째만 마스킹 하지 않음 
        is_text = masking_fn([0, 1, 2, 3, 4], position_ids)
        res_batch = jnp.where(is_asset | is_text, test_batch, masked_batch)
      elif len(img_true_indices) != 0 :                                         # image만 있는 경우는 입력 seq의 첫 번째만 마스킹 하지 않음 
        is_image = masking_fn([0, 1, 2, 3, 4], position_ids)
        res_batch = jnp.where(is_asset | is_image, test_batch, masked_batch)
      else : return jnp.where(is_asset, test_batch, masked_batch)               # 예외 case가 존재할 경우 'a' decode방식을 따름 

      return res_batch
    
    if conditional == "none":
      total_time = 0.
      for idx in range(sample_step_num):
        if use_vertical:
          test_label = jnp.full((batch_size, 1), vertical_idx)
        else:
          test_label = None
        asset_num = np.random.choice(max_asset_num, batch_size, p=prior) + 1
        rng, sample_rng = jax.random.split(rng, 2)
        p_rng = jax.random.split(sample_rng, jax.local_device_count())

        # All mask symbols.
        asset_num = jnp.array(asset_num, dtype="int32")[Ellipsis, None]
        # element_num = asset_num * 5
        element_num = asset_num * self.total_dim
        masked_batch = jnp.full((batch_size, 128), 3)

        position_ids = jnp.arange(masked_batch.shape[-1])[None, :]
        # Pads masked batch.
        masked_batch = jnp.where(position_ids >= element_num, 0, masked_batch)
        masked_batch = common_utils.shard(masked_batch)
        test_label = common_utils.shard(test_label)

        p_rng = jax.random.split(rng, jax.local_device_count())
        start_time = time.time()

        sample_layouts = p_generate_batch(masked_batch, test_label, p_rng,
                                          state)
        total_time += time.time() - start_time
        start_time = time.time()
        sample_layouts = tohost(sample_layouts)
        generated_sample_list.append(sample_layouts - offset)

      generated_samples = jnp.concatenate(generated_sample_list, axis=0)
      real_samples = None
      logging.info("decoding time: (%.4f)", total_time)
      return generated_samples, real_samples, image_link

    # 'a' : category부분만 남기고 나머지를 생성하는 decode 방식
    # 'a+s' : category와 size만 남기고 나머지 위치를 생성하는 decode 방식
    # 'custom으로 시작하는 이름' : 자세한 설명은 custom_decode 함수 참고 
    for idx, test_batch in enumerate(test_iter):
      if idx >= sample_step_num:
        break
      asset_num = np.random.choice(max_asset_num, batch_size, p=prior) + 1
      rng, sample_rng = jax.random.split(rng, 2)
      p_rng = jax.random.split(sample_rng, jax.local_device_count())
      test_batch = jax.tree_map(lambda x: x._numpy(), test_batch)  # pylint: disable=protected-access
      test_batch, _ = self.preprocess_batch(test_batch, batch_size, dataset,
                                            use_vertical)
      if test_batch is None or (conditional == "none" and
                                idx == sample_step_num):
        break
      test_batch = tohost(test_batch["targets"])
      if use_vertical:
        test_label = jnp.full((batch_size, 1), vertical_idx)
      else:
        test_label = None

      # All mask symbols.
      if conditional != "none":
        # asset_num = jnp.sum(test_batch > 0, axis=1, keepdims=True) // 5
        asset_num = jnp.sum(
            test_batch > 0, axis=1, keepdims=True) // self.total_dim
      else:
        asset_num = jnp.array(asset_num, dtype="int32")[Ellipsis, None]
      # element_num = asset_num * 5
      element_num = asset_num * self.total_dim
      masked_batch = jnp.full_like(test_batch, 3)

      position_ids = jnp.arange(masked_batch.shape[-1])[None, :]
      # is_asset = position_ids % 5 == 0
      # is_size = (position_ids % 5 == 1) | (position_ids % 5 == 2)
      is_asset = position_ids % self.total_dim == 0
      is_size = functools.reduce(lambda x, y: x | y, [
          position_ids % self.total_dim == i
          for i in range(1, self.layout_dim + 1)
      ])

      # custom decode
      dataset_name = datasets_info.DatasetName(self.config.dataset)
      label_to_id = datasets_info.get_label_to_id_map(dataset_name)
      # text id를 제외한 나머지 id를 list로 저장
      except_text_label = [value for key, value in label_to_id.items() if not key.startswith("text_")]
      # text id를 list로 저장 
      text_label = [value for key, value in label_to_id.items() if key.startswith("text_")]

      # test_batch의 real_sample에서 image에 해당하는 id가 있으면 해당 위치 index list 저장
      img_true_indices = find_indices_of_no_mask_id(test_batch, except_text_label, is_asset)
      # test_batch의 real_sample에서 text에 해당하는 id가 있으면 해당 위치 index list 저장
      txt_true_indices = find_indices_of_no_mask_id(test_batch, text_label, is_asset)

      if conditional.startswith("custom"):
        masked_batch = custom_decode( masked_batch=masked_batch,
                                      test_batch=test_batch,
                                      position_ids=position_ids,
                                      is_asset=is_asset,
                                      img_true_indices=img_true_indices,
                                      txt_true_indices=txt_true_indices)
      elif conditional == "a+s":
        masked_batch = jnp.where(is_asset | is_size, test_batch, masked_batch)
      elif conditional == "a":
        masked_batch = jnp.where(is_asset, test_batch, masked_batch)
      # Pads masked batch.
      masked_batch = jnp.where(position_ids >= element_num, 0, masked_batch)
      masked_batch = common_utils.shard(masked_batch)

      p_rng = jax.random.split(rng, jax.local_device_count())
      test_label = common_utils.shard(test_label)
      sample_layouts = p_generate_batch(masked_batch, test_label, p_rng, state)
      sample_layouts = tohost(sample_layouts)
      generated_sample_list.append(sample_layouts - offset)
      real_sample_list.append(test_batch - offset)
    generated_samples = jnp.concatenate(generated_sample_list, axis=0)
    real_samples = jnp.concatenate(real_sample_list, axis=0)
    return generated_samples, real_samples, image_link

  def sample_step(self, rng, state, model_dict, pos_info):
    """Samples layouts just for visualization during training."""
    pass

  def incremental_decode(self,
                         rng,
                         variables,
                         model,
                         pos_info,
                         batch=None,
                         label=None,
                         iterative_nums=None,
                         conditional="none",
                         logits_mask=None
                         ):
    """Feeds the inputs sequentially to decode non-autoregressively."""

    def tokens_to_logits(masked_batch):
      logits = model().apply(variables, masked_batch, label, deterministic=True)
      return logits
    seqs = layout_bert_fast_decode.decode(batch, tokens_to_logits,
                                          self.config.sampling_method, rng,
                                          logits_mask,
                                          iterative_nums=iterative_nums,
                                          layout_dim=self.layout_dim)
    return seqs

  def sample_one_batch(self, batch, label, rng, state, model_dict, pos_info,
                       iterative_num, conditional, logits_mask):
    """Samples one batch for eval."""
    model = model_dict["model"]
    variables = {"params": state.optimizer.target}
    variables.update(state.model_state)

    x = self.incremental_decode(rng, variables, model, pos_info, batch, label,
                                iterative_num, conditional,
                                logits_mask)
    return x
