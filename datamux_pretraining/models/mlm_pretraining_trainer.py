# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import os
import re
import sys
from logging import StreamHandler
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    is_fairscale_available,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset 
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_training_run_on_sagemaker,
)
from transformers.modeling_utils import PreTrainedModel 
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    nested_detach,
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    PredictionOutput,
    denumpify_detensorize,
    get_last_checkpoint,
    EvalLoopOutput
)
from transformers.utils import logging
from transformers import Trainer
from transformers.integrations import WandbCallback, rewrite_logs

from datamux_pretraining.models.utils import is_torch_tpu_available, entropy

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    import fairscale
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

    if version.parse(fairscale.__version__) >= version.parse("0.3"):
        from fairscale.nn.data_parallel import (
            FullyShardedDataParallel as FullyShardedDDP,
        )
        from fairscale.nn.wrap import auto_wrap
    else:
        FullyShardedDDP = None

import torch.distributed as dist

if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


class WandbCallbackViz(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model, reinit=False)
        is_table = len(logs) == 1

        if state.is_world_process_zero:
            if is_table:
                self._wandb.log(logs)
            else:
                logs = rewrite_logs(logs)
                self._wandb.log(logs, step=state.global_step)


class MuxedMLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        assert "train_collator" in kwargs
        assert "eval_collator" in kwargs
        self.train_data_collator = kwargs.pop("train_collator")
        self.eval_data_collator = kwargs.pop("eval_collator")
        self.data_collator = None
        super().__init__(*args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        task_loss = None
        retrieval_loss = None

        labels = inputs["labels"].clone()

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                    )
                else:
                    logits = outputs[1:]
                if "task_loss" in outputs:
                    task_loss = outputs["task_loss"].mean().detach()
                if "retrieval_loss" in outputs:
                    retrieval_loss = outputs["retrieval_loss"].mean().detach()
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]
        if prediction_loss_only:
            return (loss, None, task_loss, retrieval_loss, outputs)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, task_loss, retrieval_loss, outputs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval=None):
        if self.control.should_log and not self.control.should_evaluate:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        model_reloaded = False
        # memory metrics - must set up as early as possible
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        ):
            logger.info(f"Loading model from {resume_from_checkpoint}).")
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
                )
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info(
                "Detected the deepspeed argument but it will not be used for evaluation"
            )
        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        num_examples = (
            (num_examples // batch_size) * batch_size
            if self.args.dataloader_drop_last
            else num_examples
        )
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        task_loss_host = None
        retrieval_loss_host = None
        hidden_states_host = None
        attention_states_host = None
        hidden_states_threshold_list_host = None
        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        task_loss_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        retrieval_loss_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        histogram_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        hidden_states_threshold_list_layer = [[DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        ) for _ in range(self.model.config.num_hidden_layers)] for _ in histogram_thresholds]

        hidden_states_gatherer_list = [
            DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=batch_size
            )
            for _ in range(self.model.config.num_hidden_layers)
        ]
        attention_weights_gatherer_list = [
            DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=batch_size
            )
            for _ in range(self.model.config.num_hidden_layers)
        ]
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, SequentialDistributedSampler
            ):
                make_multiple_of = dataloader.sampler.batch_size

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]
            ).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, _, task_loss, retrieval_loss, outputs = self.prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )

                if task_loss is not None:
                    task_losses = task_loss.repeat(batch_size)
                    task_loss_host = (
                        task_losses
                        if task_loss_host is None
                        else torch.cat((task_loss_host, task_losses), dim=0)
                    )

                if retrieval_loss is not None:
                    retrieval_losses = retrieval_loss.repeat(batch_size)
                    retrieval_loss_host = (
                        retrieval_losses
                        if retrieval_loss_host is None
                        else torch.cat((retrieval_loss_host, retrieval_losses), dim=0)
                    )
                if "hidden_states" in outputs and outputs["hidden_states"] is not None:
                    cur_hidden_states = outputs["hidden_states"]
                    # ignore the first embedding layer
                    cur_hidden_states = cur_hidden_states[1:]
                    cur_hidden_states_mean = torch.stack(
                        cur_hidden_states, dim=-1
                    ).view(-1, len(cur_hidden_states))
                    cur_hidden_states_mean = torch.abs(cur_hidden_states_mean)
                    all_hidden_states_threshold_list = []
                    # histogram analysis
                    for i, threshold in enumerate(histogram_thresholds):
                        cur_hidden_states_threshold = cur_hidden_states_mean < threshold
                        cur_hidden_states_threshold = cur_hidden_states_threshold.float()
                        # calculate percentage of values less than threshold
                        cur_hidden_states_threshold = cur_hidden_states_threshold.mean(dim=0)
                        all_hidden_states_threshold_list.append(cur_hidden_states_threshold)

                    all_hidden_states_threshold_list = torch.stack(all_hidden_states_threshold_list, dim=0)
                    all_hidden_states_threshold_list = all_hidden_states_threshold_list.unsqueeze(0)
                    all_hidden_states_threshold_list = all_hidden_states_threshold_list.repeat(
                        batch_size, 1, 1
                    )                      
                    if hidden_states_threshold_list_host is not None:
                        hidden_states_threshold_list_host = torch.cat(
                            (hidden_states_threshold_list_host, all_hidden_states_threshold_list), dim=0
                        )
                    else:
                        hidden_states_threshold_list_host = all_hidden_states_threshold_list
                    
                    # mean analysis
                    cur_hidden_states_mean = torch.mean(cur_hidden_states_mean, dim=0)
                    cur_hidden_states_mean = cur_hidden_states_mean.unsqueeze(0)
                    cur_hidden_states_mean = cur_hidden_states_mean.repeat(
                        batch_size, 1
                    )
                    if hidden_states_host is not None:
                        hidden_states_host = torch.cat(
                            (hidden_states_host, cur_hidden_states_mean), dim=0
                        )
                    else:
                        hidden_states_host = cur_hidden_states_mean
                if "attentions" in outputs and outputs["attentions"] is not None:
                    cur_attention_states = outputs["attentions"]
                    cur_attention_states_stacked = torch.stack(
                        cur_attention_states, dim=1
                    )
                    # entropy function normalizes across the last dimension
                    cur_attention_states_entropy = entropy(cur_attention_states_stacked)
                    cur_attention_states_entropy_permuted = torch.permute(
                        cur_attention_states_entropy, (0, 2, 3, 1)
                    )
                    cur_attention_states_mean = torch.mean(
                        cur_attention_states_entropy_permuted.reshape(
                            -1, len(cur_attention_states)
                        ),
                        dim=0,
                    )

                    cur_attention_states_mean = cur_attention_states_mean.unsqueeze(0)
                    cur_attention_states_mean = cur_attention_states_mean.repeat(
                        batch_size, 1
                    )
                    if attention_states_host is not None:
                        attention_states_host = torch.cat(
                            (attention_states_host, cur_attention_states_mean), dim=0
                        )
                    else:
                        attention_states_host = cur_attention_states_mean
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                eval_losses_gatherer.add_arrays(
                    self._gather_and_numpify(losses_host, "eval_losses")
                )
                if task_loss_host is not None:
                    task_loss_gatherer.add_arrays(
                        self._gather_and_numpify(task_loss_host, "eval_task_losses")
                    )
                if retrieval_loss_host is not None:
                    retrieval_loss_gatherer.add_arrays(
                        self._gather_and_numpify(
                            retrieval_loss_host, "eval_retrieval_losses"
                        )
                    )
                if hidden_states_host is not None:
                    for i, hidden_states_gatherer in enumerate(
                        hidden_states_gatherer_list
                    ):
                        hidden_states_gatherer.add_arrays(
                            self._gather_and_numpify(
                                hidden_states_host[:, i], f"eval_hidden_states_{i}"
                            )
                        )
                if hidden_states_threshold_list_host is not None:
                    for threshold_id, threshold_gatherer_list in enumerate(
                        hidden_states_threshold_list_layer 
                    ):
                        for layer_id, cur_gatherer in enumerate(threshold_gatherer_list):
                            cur_gatherer.add_arrays(
                                self._gather_and_numpify(
                                    hidden_states_threshold_list_host[:, threshold_id, layer_id], f"eval_hidden_states_threshold_{histogram_thresholds[threshold_id]}_{layer_id}"
                                )
                            )
                if attention_states_host is not None:
                    for i, attention_weights_gatherer in enumerate(
                        attention_weights_gatherer_list
                    ):
                        attention_weights_gatherer.add_arrays(
                            self._gather_and_numpify(
                                attention_states_host[:, i],
                                f"eval_attention_weights_{i}",
                            )
                        )
                # Set back to None to begin a new accumulation
                losses_host, task_loss_host, retrieval_loss_host, hidden_states_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses")
        )

        if task_loss_host is not None:
            task_loss_gatherer.add_arrays(
                self._gather_and_numpify(task_loss_host, "eval_lm_losses")
            )
        if retrieval_loss_host is not None:
            retrieval_loss_gatherer.add_arrays(
                self._gather_and_numpify(retrieval_loss_host, "eval_retrieval_losses")
            )
        if hidden_states_host is not None:
            hidden_states_host_splits = torch.split(hidden_states_host, 1, dim=1)
            for i, hidden_states_gatherer in enumerate(hidden_states_gatherer_list):
                hidden_states_gatherer.add_arrays(
                    self._gather_and_numpify(
                        hidden_states_host_splits[i].squeeze(1).clone(),
                        f"eval_hidden_states_{i}",
                    )
                )
        if hidden_states_threshold_list_host is not None:
            for threshold_id, threshold_gatherer_list in enumerate(
                hidden_states_threshold_list_layer 
            ):
                for layer_id, cur_gatherer in enumerate(threshold_gatherer_list):
                    cur_gatherer.add_arrays(
                        self._gather_and_numpify(
                            hidden_states_threshold_list_host[:, threshold_id, layer_id], f"eval_hidden_states_threshold_{histogram_thresholds[threshold_id]}_{layer_id}"
                        )
                    )
        if attention_states_host is not None:
            attention_states_host_splits = torch.split(attention_states_host, 1, dim=1)
            for i, attention_weights_gatherer in enumerate(
                attention_weights_gatherer_list
            ):
                attention_weights_gatherer.add_arrays(
                    self._gather_and_numpify(
                        attention_states_host_splits[i].squeeze(1).clone(),
                        f"eval_attention_weights_{i}",
                    )
                )
                # Set back to None to begin a new accumulation
        eval_loss = eval_losses_gatherer.finalize()
        task_loss = (
            task_loss_gatherer.finalize() if task_loss_host is not None else None
        )
        retrieval_loss = (
            retrieval_loss_gatherer.finalize()
            if retrieval_loss_host is not None
            else None
        )
        if hidden_states_host is not None:
            hidden_states_list = []
            for hidden_states_gatherer in hidden_states_gatherer_list:
                hidden_states_list.append(hidden_states_gatherer.finalize())
            hidden_states_host = np.stack(hidden_states_list, axis=1)

        if attention_states_host is not None:
            attention_weights_list = []
            for attention_weights_gatherer in attention_weights_gatherer_list:
                attention_weights_list.append(attention_weights_gatherer.finalize())
            attention_states_host = np.stack(attention_weights_list, axis=1)
        if hidden_states_threshold_list_host is not None:
            hidden_states_threshold_list = []
            for threshold_gatherer_list in hidden_states_threshold_list_layer:
                cur_threshold_list = []
                for layer_id, cur_gatherer in enumerate(threshold_gatherer_list):
                    cur_threshold_list.append(
                        cur_gatherer.finalize()
                    )
                hidden_states_threshold_list.append(np.stack(cur_threshold_list, axis=1))
            hidden_states_threshold_list_host = np.stack(hidden_states_threshold_list, axis=1)
        metrics = {}
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        if eval_loss is not None:
            eval_loss = eval_loss[eval_loss != -100]
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()
            if task_loss is not None:
                task_loss = task_loss[task_loss != -100]
                metrics[f"{metric_key_prefix}_mlm_loss"] = task_loss.mean().item()
            if retrieval_loss is not None:
                retrieval_loss = retrieval_loss[retrieval_loss != -100]
                metrics[
                    f"{metric_key_prefix}_retrieval_loss"
                ] = retrieval_loss.mean().item()
            if hidden_states_host is not None:
                for i in range(hidden_states_host.shape[1]):
                    cur_hid = hidden_states_host[:, i]
                    cur_hid = cur_hid[cur_hid != -100]
                    metrics[
                        f"{metric_key_prefix}_hidden_states_mean_{i}"
                    ] = cur_hid.mean().item()
            if attention_states_host is not None:
                for i in range(attention_states_host.shape[1]):
                    cur_attn = attention_states_host[:, i]
                    cur_attn = cur_attn[cur_attn != -100]
                    metrics[
                        f"{metric_key_prefix}_attention_weights_mean_{i}"
                    ] = cur_attn.mean().item()
            if hidden_states_threshold_list_host is not None:
                for threshold_id in range(len(histogram_thresholds)):
                    for layer_id in range(hidden_states_threshold_list_host.shape[2]):
                        cur_val = hidden_states_threshold_list_host[:, threshold_id, layer_id]
                        cur_val = cur_val[cur_val != -100]
                        metrics[
                            f"{metric_key_prefix}_hidden_states_threshold_{histogram_thresholds[threshold_id]}_{layer_id}"
                        ] = cur_val.mean().item()
            metrics = denumpify_detensorize(metrics)
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        if not isinstance(self.eval_dataset, IterableDataset):
            num_samples = len(self.eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(self.eval_dataset, IterableDatasetShard) and hasattr(self.eval_dataset, "num_examples"):
            num_samples = self.eval_dataset.num_examples
        else:
            raise ValueError(
                f"Trainer: the eval_dataset {self.eval_dataset} should either be a simple dataset or an IterableDatasetShard."
            )
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )