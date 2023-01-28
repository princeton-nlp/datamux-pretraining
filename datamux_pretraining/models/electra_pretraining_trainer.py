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
from dis import dis
import gc
import inspect
import math
import os
import itertools 
import re
import shutil
import sys
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torchmetrics
import wandb
import warnings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
    is_fairscale_available,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_training_run_on_sagemaker,
)
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    nested_concat,
    nested_detach,
)
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    ShardedDDPOption,
    TrainOutput,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.utils import logging
from transformers import Trainer
from transformers.integrations import WandbCallback, TensorBoardCallback, rewrite_logs

from datamux_pretraining.models.utils import is_torch_tpu_available
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback
TRAINING_ARGS_NAME = "training_args.bin"

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


class TensorBoardCallbackElectra(TensorBoardCallback):
    def on_train_begin(self, args, state, control, **kwargs):

        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            discriminator_model_config_json = None
            generator_model_config_json = None
            if "model" in kwargs:
                model = kwargs["model"]
                if (
                    hasattr(model, "discriminator_config")
                    and model.discriminator_config is not None
                ):
                    discriminator_model_config_json = model.discriminator_config
                    self.tb_writer.add_text(
                        "discriminator_model_config",
                        discriminator_model_config_json.to_json_string(),
                    )
                if (
                    hasattr(model, "generator_config")
                    and model.generator_config is not None
                ):
                    generator_model_config_json = model.generator_config
                    self.tb_writer.add_text(
                        "generator_model_config",
                        generator_model_config_json.to_json_string(),
                    )
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                if discriminator_model_config_json is None:
                    discriminator_model_config_json = {}
                else:
                    discriminator_model_config_json = (
                        discriminator_model_config_json.to_dict()
                    )
                if generator_model_config_json is None:
                    generator_model_config_json = {}
                else:
                    generator_model_config_json = generator_model_config_json.to_dict()
                hparams = dict(
                    args.to_sanitized_dict(), **discriminator_model_config_json
                )
                hparams.update(generator_model_config_json)
                valid_types = [bool, int, float, str, torch.Tensor]
                sanitized_hparams = {
                    k: v if type(v) in valid_types else str(v)
                    for k, v in hparams.items()
                }
                self.tb_writer.add_hparams(
                    sanitized_hparams, metric_dict={"accuracy": 0}
                )


class MuxedElectraTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        self.table_viz = kwargs.pop("table_viz", True)
        assert "train_collator" in kwargs
        assert "eval_collator" in kwargs
        self.train_data_collator = kwargs.pop("train_collator")
        self.eval_data_collator = kwargs.pop("eval_collator")
        self.num_instances = kwargs.pop("num_instances")
        self.data_collator = None
        super().__init__(*args, **kwargs)
        self.pop_callback(WandbCallback)
        # self.add_callback(WandbCallbackViz)
        self.pop_callback(TensorBoardCallback)
        self.add_callback(TensorBoardCallbackElectra)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
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
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        activation_analysis = False
        if activation_analysis:
            # go through the eval dataset and get the activations
            # for each layer, average the hidden states and the attentions
            # and store in each result
            pass
        run_qual_report = False 
        if run_qual_report:
            num_anchors = self.model.discriminator_config.num_instances
            perms = list(itertools.permutations(range(num_anchors)))
            perms = np.array(perms)
            perms = torch.from_numpy(perms).to(self.args.device)
            perms = torch.permute(perms, (1, 0))
            eval_dataloader =  self.get_train_dataloader()
            model = self._wrap_model(self.model, training=False)
            model.eval()
            anchors = None
            anchors_labels = None
            anchors_attention_mask = None
            anchor_representations = {i: [] for i in range(num_anchors)}
            with torch.no_grad():
                for _, inputs in enumerate(tqdm(eval_dataloader)):
                    inputs = self._prepare_inputs(inputs)
                    if anchors is None:
                        anchors = inputs["input_ids"][:num_anchors]
                        anchors_labels = inputs["labels"][:num_anchors]
                        anchors_attention_mask = inputs["attention_mask"][:num_anchors]
                        break
            permuted_anchor_data = anchors[perms]
            permuted_anchor_labels = anchors_labels[perms]
            permuted_anchor_attention_mask = anchors_attention_mask[perms]
            with torch.no_grad():
                for perm_id in tqdm(range(perms.shape[1])):
                    inputs = {}
                    inputs["input_ids"] = permuted_anchor_data[:, perm_id]
                    inputs["labels"] = permuted_anchor_labels[:, perm_id]
                    inputs["attention_mask"] = permuted_anchor_attention_mask[:, perm_id]
                    cur_perms = perms[:, perm_id]
                    inputs["return_dict"] = True
                    outputs = model(**inputs)
                    outputs["hidden_states"] = outputs["hidden_states"][:, 0, :]
                    for hidden_rep_id in range(num_anchors):
                        anchor_representations[cur_perms[hidden_rep_id].item()].append(outputs.hidden_states[hidden_rep_id])
            # t-sne plot
            anchor_representations_stacked = []
            for anchor_id in range(num_anchors):
               anchor_representations_stacked.append(torch.stack(anchor_representations[anchor_id]))
            anchor_representations_stacked = torch.cat(anchor_representations_stacked)
            anchor_representations_stacked = anchor_representations_stacked.cpu().numpy() 
            average_cos_similarity = 0 
            for anchor_i in range(num_anchors):
                for anchor_j in range(anchor_i +1, num_anchors):
                    anchor_i_representations = torch.stack(anchor_representations[anchor_i])
                    anchor_j_representations = torch.stack(anchor_representations[anchor_j])
                    average_cos_similarity += (torch.matmul(anchor_i_representations, anchor_j_representations.t()) / (torch.norm(anchor_i_representations, dim=1) * torch.norm(anchor_j_representations, dim=1))).mean()
            average_cos_similarity /= (num_anchors * (num_anchors - 1) * 0.5)
            pca_50 = PCA(n_components=200)
            pca_result_50 = pca_50.fit_transform(anchor_representations_stacked)
            tsne = TSNE(n_components=2, verbose=1, n_iter=500)
            tsne_pca_results = tsne.fit_transform(pca_result_50)
            df = pd.DataFrame()
            df["tsne_1"] = tsne_pca_results[:, 0]
            df["tsne_2"] = tsne_pca_results[:, 1]
            df["sample"] = np.repeat(np.arange(num_anchors), len(anchor_representations[0]))
            df["sample"] = df["sample"].apply(lambda i: str(i))
            sns.scatterplot(
            x="tsne_1", y="tsne_2",
            hue='sample',
            palette=sns.color_palette("bright", num_anchors),
            data=df,
            legend="full",
            alpha=0.3,
            )
            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.title(f'Permutation analysis: N = {self.model.discriminator_config.num_instances}', fontsize=16)
            plt.legend(loc="lower right", fontsize=8)
            plt.savefig(f"permutation_fig_{self.model.discriminator_config.num_instances}.png")
            df.to_csv(f"permutation_fig_{self.model.discriminator_config.num_instances}.csv")
            print(f"average cos similarity: {average_cos_similarity}")
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        wandb_table=None,
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

        retrieval_loss = None

        with torch.no_grad():
            if has_labels:
                (
                    loss,
                    disc_loss,
                    gen_loss,
                    retrieval_loss,
                    gen_retrieval_loss,
                    gen_mlm_loss,
                    disc_logits,
                    mlm_gen_logits,
                    sampled_input_ids,
                    discriminator_labels,
                    head_weights,
                    head_accuracies,
                    outputs,
                ) = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if disc_loss is not None:
                    disc_loss = disc_loss.mean().detach()
                if gen_loss is not None:
                    gen_loss = gen_loss.mean().detach()
                if retrieval_loss is not None:
                    retrieval_loss = retrieval_loss.mean().detach()
                if gen_retrieval_loss is not None:
                    gen_retrieval_loss = gen_retrieval_loss.mean().detach()
                if gen_mlm_loss is not None:
                    gen_mlm_loss = gen_mlm_loss.mean().detach()
                if disc_logits is not None:
                    logits = disc_logits.detach()
                else:
                    logits = None
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

        # calculate discriminator metrics
        # calculate disc. metrics
        disc_pred_labels = None
        disc_accuracy = None
        disc_f1 = None
        disc_precision = None
        disc_auc = None

        # calculate discriminator metrics
        # calculate disc. metrics
        if disc_logits is not None:
            disc_probs = torch.sigmoid(disc_logits).detach().view(-1)
            disc_pred_labels = (disc_probs > 0.5).long()
            discriminator_labels  = discriminator_labels.reshape(-1)
            disc_pred_labels  = disc_pred_labels.view(-1)
            disc_accuracy = torch.mean(
                (disc_pred_labels == discriminator_labels).float().detach()
            )
            disc_f1 = torchmetrics.functional.f1(disc_pred_labels, discriminator_labels, multiclass=False).detach()
            disc_precision = torchmetrics.functional.precision(disc_pred_labels, discriminator_labels, multiclass=False).detach()
            disc_auc = torch.zeros_like(disc_precision).detach()
        if wandb_table is not None:

            input_ids = sampled_input_ids.clone()
            labels = discriminator_labels.clone()
            batch_size, seq_length = input_ids.size()
            num_instances = self.num_instances
            # decode all sentences in the batch
            tokenizer = self.tokenizer

            disc_pred_labels = disc_pred_labels.view(batch_size, seq_length)

            def replace_tags(string):
                return string.replace(">", "]").replace("<", "[")

            if batch_size % num_instances == 0:
                for batch_id in range((batch_size // num_instances) * num_instances):
                    cur_sampled_masked_indices = discriminator_labels[batch_id]
                    masked_positions = cur_sampled_masked_indices.nonzero(
                        as_tuple=True
                    )[0]
                    for masked_pos in masked_positions:
                        # add a new entry to the table
                        cur_sentences = [
                            replace_tags(
                                tokenizer.decode(
                                    inputs["input_ids"][batch_id][
                                        : max(masked_pos - 10, 0)
                                    ]
                                )
                            )
                            + " <strong> %s </strong> "
                            % replace_tags(
                                tokenizer.decode(
                                    inputs["input_ids"][batch_id][
                                        max(masked_pos - 10, 0) : min(
                                            masked_pos + 10, seq_length - 1
                                        )
                                    ]
                                )
                            )
                            + replace_tags(
                                tokenizer.decode(
                                    inputs["input_ids"][batch_id][
                                        min(masked_pos + 10, seq_length - 1) :
                                    ]
                                )
                            )
                        ]
                        cur_snippet = (
                            replace_tags(
                                tokenizer.decode(
                                    input_ids[batch_id][
                                        max(masked_pos - 10, 0) : masked_pos
                                    ]
                                )
                            )
                            + " <strong> %s </strong> "
                            % replace_tags(
                                tokenizer.decode(
                                    input_ids[batch_id][masked_pos : masked_pos + 1]
                                )
                            )
                            + replace_tags(
                                tokenizer.decode(
                                    input_ids[batch_id][
                                        min(masked_pos + 1, seq_length - 1) : masked_pos
                                        + 10
                                    ]
                                )
                            )
                        )

                        try:
                            cur_sentences_html = wandb.Html(
                                " <p> %s </p> " % " <br> <br> ".join(cur_sentences)
                            )
                            cur_snippet_html = wandb.Html(" <p> %s </p> " % cur_snippet)

                        except:
                            cur_sentences_html = wandb.Html("<p>  </p>")
                            cur_snippet_html = wandb.Html("<p>  </p>")

                        wandb_table.add_data(
                            cur_sentences_html,
                            cur_snippet_html,
                            disc_pred_labels[batch_id][masked_pos].item(),
                            True,
                        )

        if prediction_loss_only:
            return (
                loss,
                None,
                None,
                retrieval_loss,
            )

        if logits is not None:
            logits = nested_detach(logits)

        return (
            loss,
            logits,
            labels,
            disc_loss,
            gen_loss,
            retrieval_loss,
            gen_retrieval_loss,
            gen_mlm_loss,
            disc_f1,
            disc_precision,
            disc_accuracy,
            disc_auc,
            head_weights,
            head_accuracies,
        )

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        tr_disc_loss,
        tr_gen_loss,
        tr_retrieval_loss,
        tr_gen_retrieval_loss,
        tr_gen_mlm_loss,
        tr_disc_accuracy,
        tr_disc_f1,
        tr_disc_precision,
        tr_disc_auc,
        tr_head_weights,
        tr_head_accuracies,
        model,
        trial,
        epoch,
    ):
        if self.control.should_log and not self.control.should_evaluate:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            disc_loss_scalar = tr_disc_loss.item() if tr_disc_loss is not None else None
            gen_loss_scalar = tr_gen_loss.item() if tr_gen_loss is not None else None
            disc_f1_scalar = tr_disc_f1.item() if tr_disc_f1 is not None else None
            disc_accuracy_scalar = (
                tr_disc_accuracy.item() if tr_disc_accuracy is not None else None
            )
            disc_precision_scalar = (
                tr_disc_precision.item() if tr_disc_precision is not None else None
            )
            disc_auc_scalar = tr_disc_auc.item() if tr_disc_auc is not None else None
            retrieval_loss_scalar = (
                tr_retrieval_loss.item() if tr_retrieval_loss is not None else None
            )
            gen_retrieval_loss_scalar = (
                tr_gen_retrieval_loss.item()
                if tr_gen_retrieval_loss is not None
                else None
            )
            gen_mlm_loss_scalar = (
                tr_gen_mlm_loss.item() if tr_gen_mlm_loss is not None else None
            )
            head_weights = (
                tr_head_weights.tolist() if tr_head_weights is not None else None
            )
            head_accuracies = (
                tr_head_accuracies.tolist() if tr_head_accuracies is not None else None
            )
            # reset tr_loss to zero
            tr_loss -= tr_loss
            tr_disc_loss -= tr_disc_loss
            tr_gen_loss -= tr_gen_loss
            tr_retrieval_loss -= tr_retrieval_loss
            tr_gen_retrieval_loss -= tr_gen_retrieval_loss
            tr_gen_mlm_loss -= tr_gen_mlm_loss
            tr_disc_accuracy -= tr_disc_accuracy
            tr_disc_f1 -= tr_disc_f1
            tr_disc_precision -= tr_disc_precision
            tr_disc_auc -= tr_disc_auc
            tr_head_weights -= tr_head_weights
            tr_head_accuracies -= tr_head_accuracies

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if disc_loss_scalar is not None:
                logs["disc_loss"] = round(
                    disc_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )

            if gen_loss_scalar is not None:
                logs["gen_loss"] = round(
                    gen_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )

            if disc_accuracy_scalar is not None:
                logs["disc_accuracy"] = round(
                    disc_accuracy_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )

            if disc_f1_scalar is not None:
                logs["disc_f1"] = round(
                    disc_f1_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )

            if disc_precision_scalar is not None:
                logs["disc_precision"] = round(
                    disc_precision_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )

            if disc_auc_scalar is not None:
                logs["disc_auc"] = round(
                    disc_auc_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )

            if retrieval_loss_scalar is not None:
                logs["retrieval_loss"] = round(
                    retrieval_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
            if gen_retrieval_loss_scalar is not None:
                logs["gen_retrieval_loss"] = round(
                    gen_retrieval_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
            if gen_mlm_loss_scalar is not None:
                logs["gen_mlm_loss"] = round(
                    gen_mlm_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
            if head_weights is not None:
                for head in range(self.num_instances):
                    logs[f"head_weights_{head}"] = head_weights[head]
            if head_accuracies is not None:
                for head in range(self.num_instances):
                    logs[f"head_accuracies_{head}"] = head_accuracies[head]

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

        # find mean eval statistics
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        disc_loss_host = None
        gen_loss_host = None
        retrieval_loss_host = None
        gen_retrieval_loss_host = None
        gen_mlm_loss_host = None
        disc_f1_host = None
        disc_precision_host = None
        disc_accuracy_host = None
        disc_auc_host = None
        head_weights_host = None
        head_accuracies_host = None

        world_size = max(1, self.args.world_size)
        eval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )

        disc_loss_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        gen_loss_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        retrieval_loss_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        gen_retrieval_loss_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        gen_mlm_loss_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        disc_f1_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        disc_precision_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        disc_accuracy_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        disc_auc_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        head_weights_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        head_accuracies_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )

        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, SequentialDistributedSampler
            ):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )
            labels_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]
            ).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader
        column_names = ["all context", "snippet", "prediction", "GT Label"]
        table = wandb.Table(columns=column_names)

        for step, inputs in enumerate(dataloader):

            (
                loss,
                logits,
                labels,
                disc_loss,
                gen_loss,
                retrieval_loss,
                gen_retrieval_loss,
                gen_mlm_loss,
                disc_f1,
                disc_precision,
                disc_accuracy,
                disc_auc,
                head_weights,
                head_accuracies,
            ) = self.prediction_step(
                model, inputs, False, ignore_keys=ignore_keys, wandb_table=None
            )
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
                if retrieval_loss is not None:
                    retrieval_losses = retrieval_loss.repeat(batch_size)
                    retrieval_loss_host = (
                        retrieval_losses
                        if retrieval_loss_host is None
                        else torch.cat((retrieval_loss_host, retrieval_losses), dim=0)
                    )
                if disc_loss is not None:
                    disc_losses = disc_loss.repeat(batch_size)
                    disc_loss_host = (
                        disc_losses
                        if disc_loss_host is None
                        else torch.cat((disc_loss_host, disc_losses), dim=0)
                    )
                if gen_loss is not None:
                    gen_losses = gen_loss.repeat(batch_size)
                    gen_loss_host = (
                        gen_losses
                        if gen_loss_host is None
                        else torch.cat((gen_loss_host, gen_losses), dim=0)
                    )
                if gen_retrieval_loss is not None:
                    gen_retrieval_losses = gen_retrieval_loss.repeat(batch_size)
                    gen_retrieval_loss_host = (
                        gen_retrieval_losses
                        if gen_retrieval_loss_host is None
                        else torch.cat(
                            (gen_retrieval_loss_host, gen_retrieval_losses), dim=0
                        )
                    )
                if gen_mlm_loss is not None:
                    gen_mlm_losses = gen_mlm_loss.repeat(batch_size)
                    gen_mlm_loss_host = (
                        gen_mlm_losses
                        if gen_mlm_loss_host is None
                        else torch.cat((gen_mlm_loss_host, gen_mlm_losses), dim=0)
                    )
            if disc_f1 is not None:
                disc_f1s = disc_f1.repeat(batch_size)
                disc_f1_host = (
                    disc_f1s
                    if disc_f1_host is None
                    else torch.cat((disc_f1_host, disc_f1s), dim=0)
                )

            if disc_auc is not None:
                disc_aucs = disc_auc.repeat(batch_size)
                disc_auc_host = (
                    disc_aucs
                    if disc_auc_host is None
                    else torch.cat((disc_auc_host, disc_aucs), dim=0)
                )

            if disc_precision is not None:
                disc_precisions = disc_precision.repeat(batch_size)
                disc_precision_host = (
                    disc_precisions
                    if disc_precision_host is None
                    else torch.cat((disc_precision_host, disc_precisions), dim=0)
                )

            if disc_accuracy is not None:
                disc_accuracies = disc_accuracy.repeat(batch_size)
                disc_accuracy_host = (
                    disc_accuracies
                    if disc_accuracy_host is None
                    else torch.cat((disc_accuracy_host, disc_accuracies), dim=0)
                )
            if head_weights is not None:
                head_weights = head_weights.repeat(batch_size)
                head_weights_host = (
                    head_weights
                    if head_weights_host is None
                    else torch.cat((head_weights_host, head_weights), dim=0)
                )

            if head_accuracies is not None:
                head_accuracies = head_accuracies.repeat(batch_size)
                head_weights_host = (
                    head_weights
                    if head_weights_host is None
                    else torch.cat((head_weights_host, head_weights), dim=0)
                )

            if logits is not None:
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )

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

                if retrieval_loss_host is not None:
                    retrieval_loss_gatherer.add_arrays(
                        self._gather_and_numpify(
                            retrieval_loss_host, "eval_retrieval_losses"
                        )
                    )

                if disc_loss_host is not None:
                    disc_loss_gatherer.add_arrays(
                        self._gather_and_numpify(disc_loss_host, "eval_disc_losses")
                    )

                if gen_loss_host is not None:
                    gen_loss_gatherer.add_arrays(
                        self._gather_and_numpify(gen_loss_host, "eval_gen_losses")
                    )

                if gen_retrieval_loss_host is not None:
                    gen_retrieval_loss_gatherer.add_arrays(
                        self._gather_and_numpify(
                            gen_retrieval_loss_host, "eval_gen_retrieval_losses"
                        )
                    )
                if gen_mlm_loss_host is not None:
                    gen_mlm_loss_gatherer.add_arrays(
                        self._gather_and_numpify(
                            gen_mlm_loss_host, "eval_gen_mlm_losses"
                        )
                    )
                if disc_f1_host is not None:
                    disc_f1_gatherer.add_arrays(
                        self._gather_and_numpify(disc_f1_host, "eval_disc_f1")
                    )

                if disc_precision_host is not None:
                    disc_precision_gatherer.add_arrays(
                        self._gather_and_numpify(
                            disc_precision_host, "eval_disc_precision"
                        )
                    )

                if disc_accuracy_host is not None:
                    disc_accuracy_gatherer.add_arrays(
                        self._gather_and_numpify(
                            disc_accuracy_host, "eval_disc_accuracy"
                        )
                    )

                if disc_auc_host is not None:
                    disc_auc_gatherer.add_arrays(
                        self._gather_and_numpify(disc_auc_host, "eval_disc_auc")
                    )

                if not prediction_loss_only:
                    preds_gatherer.add_arrays(
                        self._gather_and_numpify(preds_host, "eval_preds")
                    )
                    labels_gatherer.add_arrays(
                        self._gather_and_numpify(labels_host, "eval_label_ids")
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None
                (
                    disc_loss_host,
                    gen_loss_host,
                    retrieval_loss_host,
                    gen_retrieval_loss_host,
                    gen_mlm_loss_host,
                    disc_f1_host,
                    disc_precision_host,
                    disc_accuracy_host,
                    disc_auc_host,
                ) = (None, None, None, None, None, None, None, None, None)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        self.callback_handler.on_log(
            self.args, self.state, self.control, {str(self.state.global_step): table}
        )
        # Gather all remaining tensors and put them back on the CPU

        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses")
        )

        if retrieval_loss_host is not None:
            retrieval_loss_gatherer.add_arrays(
                self._gather_and_numpify(retrieval_loss_host, "eval_retrieval_losses")
            )

        if gen_retrieval_loss_host is not None:
            gen_retrieval_loss_gatherer.add_arrays(
                self._gather_and_numpify(
                    gen_retrieval_loss_host, "eval_gen_retrieval_losses"
                )
            )

        if gen_mlm_loss_host is not None:
            gen_mlm_loss_gatherer.add_arrays(
                self._gather_and_numpify(gen_mlm_loss_host, "eval_gen_mlm_losses")
            )
        if disc_loss_host is not None:
            disc_loss_gatherer.add_arrays(
                self._gather_and_numpify(disc_loss_host, "eval_disc_losses")
            )

        if gen_loss_host is not None:
            gen_loss_gatherer.add_arrays(
                self._gather_and_numpify(gen_loss_host, "eval_gen_losses")
            )

        if disc_f1_host is not None:
            disc_f1_gatherer.add_arrays(
                self._gather_and_numpify(disc_f1_host, "eval_disc_f1")
            )

        if disc_precision_host is not None:
            disc_precision_gatherer.add_arrays(
                self._gather_and_numpify(disc_precision_host, "eval_disc_precision")
            )

        if disc_accuracy_host is not None:
            disc_accuracy_gatherer.add_arrays(
                self._gather_and_numpify(disc_accuracy_host, "eval_disc_accuracy")
            )

        if disc_auc_host is not None:
            disc_auc_gatherer.add_arrays(
                self._gather_and_numpify(disc_auc_host, "eval_disc_auc")
            )

        if not prediction_loss_only:
            preds_gatherer.add_arrays(
                self._gather_and_numpify(preds_host, "eval_preds")
            )
            labels_gatherer.add_arrays(
                self._gather_and_numpify(labels_host, "eval_label_ids")
            )
        eval_loss = eval_losses_gatherer.finalize()
        retrieval_loss = (
            retrieval_loss_gatherer.finalize()
            if retrieval_loss_host is not None
            else None
        )
        disc_loss = (
            disc_loss_gatherer.finalize() if disc_loss_host is not None else None
        )
        gen_loss = gen_loss_gatherer.finalize() if gen_loss_host is not None else None
        gen_retrieval_loss = (
            gen_retrieval_loss_gatherer.finalize()
            if gen_retrieval_loss_host is not None
            else None
        )
        gen_mlm_loss = (
            gen_mlm_loss_gatherer.finalize() if gen_mlm_loss_host is not None else None
        )
        disc_f1 = disc_f1_gatherer.finalize() if disc_f1_host is not None else None
        disc_precision = (
            disc_precision_gatherer.finalize()
            if disc_precision_host is not None
            else None
        )
        disc_accuracy = (
            disc_accuracy_gatherer.finalize()
            if disc_accuracy_host is not None
            else None
        )
        disc_auc = disc_auc_gatherer.finalize() if disc_auc_host is not None else None
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )
        else:
            metrics = {}
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)
        if eval_loss is not None:
            eval_loss = eval_loss[eval_loss != -100]
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()
            if retrieval_loss is not None:
                retrieval_loss = retrieval_loss[retrieval_loss != -100]
                metrics[
                    f"{metric_key_prefix}_retrieval_loss"
                ] = retrieval_loss.mean().item()
            if disc_loss is not None:
                disc_loss = disc_loss[disc_loss != -100]
                metrics[f"{metric_key_prefix}_disc_loss"] = disc_loss.mean().item()
            if gen_loss is not None:
                gen_loss = gen_loss[gen_loss != -100]
                metrics[f"{metric_key_prefix}_gen_loss"] = gen_loss.mean().item()
            if gen_retrieval_loss is not None:
                gen_retrieval_loss = gen_retrieval_loss[gen_retrieval_loss != -100]
                metrics[
                    f"{metric_key_prefix}_gen_retrieval_loss"
                ] = gen_retrieval_loss.mean().item()
            if gen_mlm_loss is not None:
                gen_mlm_loss = gen_mlm_loss[gen_mlm_loss != -100]
                metrics[
                    f"{metric_key_prefix}_gen_mlm_loss"
                ] = gen_mlm_loss.mean().item()
            if disc_f1 is not None:
                disc_f1 = disc_f1[disc_f1 != -100]
                metrics[f"{metric_key_prefix}_disc_f1"] = disc_f1.mean().item()
            if disc_precision is not None:
                disc_precision = disc_precision[disc_precision != -100]
                metrics[
                    f"{metric_key_prefix}_disc_precision"
                ] = disc_precision.mean().item()
            if disc_accuracy is not None:
                disc_accuracy = disc_accuracy[disc_accuracy != -100]
                metrics[
                    f"{metric_key_prefix}_disc_accuracy"
                ] = disc_accuracy.mean().item()
            if disc_auc is not None:
                disc_auc = disc_auc[disc_auc != -100]
                metrics[f"{metric_key_prefix}_disc_auc"] = disc_auc.mean().item()
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        elif is_datasets_available() and isinstance(
            self.train_dataset, datasets.Dataset
        ):
            self._remove_unused_columns(self.train_dataset, description="training")
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
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(
            eval_dataset, datasets.Dataset
        ):
            self._remove_unused_columns(eval_dataset, description="evaluation")
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
        elif is_datasets_available() and isinstance(
            test_dataset, datasets.Dataset
        ):
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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs["return_dict"] = True
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            disc_loss = outputs["disc_loss"] if "disc_loss" in outputs else None
            gen_loss = outputs["gen_loss"] if "gen_loss" in outputs else None
            retrieval_loss = (
                outputs["retrieval_loss"] if "retrieval_loss" in outputs else None
            )
            gen_retrieval_loss = (
                outputs["gen_retrieval_loss"] if "gen_retrieval_loss" in outputs else None)
            gen_mlm_loss = outputs["gen_mlm_loss"] if "gen_mlm_loss" in outputs else None
            disc_logits = outputs["disc_logits"] if "disc_logits" in outputs else None
            mlm_gen_logits = (
                outputs["mlm_gen_logits"] if "mlm_gen_logits" in outputs else None
            )
            sampled_input_ids = (
                outputs["sampled_input_ids"] if "sampled_input_ids" in outputs else None
            )
            discriminator_labels = (
                outputs["corruption_applied"]
                if "corruption_applied" in outputs
                else None
            )
            head_weights = (
                outputs["head_weights"] if "head_weights" in outputs else None
            )
            head_accuracies = (
                outputs["train_metrics_per_head"]
                if "train_metrics_per_head" in outputs
                else None
            )

        return (
            (
                loss,
                disc_loss,
                gen_loss,
                retrieval_loss,
                gen_retrieval_loss,
                gen_mlm_loss,
                disc_logits,
                mlm_gen_logits,
                sampled_input_ids,
                discriminator_labels,
                head_weights,
                head_accuracies,
                outputs,
            )
            if return_outputs
            else (
                loss,
                disc_loss,
                gen_loss,
                retrieval_loss,
                disc_logits,
                mlm_gen_logits,
                sampled_input_ids,
                discriminator_labels,
            )
        )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)
        if self.use_amp:
            with autocast():
                (
                    loss,
                    disc_loss,
                    gen_loss,
                    retrieval_loss,
                    gen_retrieval_loss,
                    gen_mlm_loss,
                    disc_logits,
                    mlm_gen_logits,
                    sampled_input_ids,
                    discriminator_labels,
                    head_weights,
                    head_accuracies,
                    outputs,
                ) = self.compute_loss(model, inputs, return_outputs=True)
        else:
            (
                loss,
                disc_loss,
                gen_loss,
                retrieval_loss,
                gen_retrieval_loss,
                gen_mlm_loss,
                disc_logits,
                mlm_gen_logits,
                sampled_input_ids,
                discriminator_labels,
                head_weights,
                head_accuracies,
                outputs,
            ) = self.compute_loss(model, inputs, return_outputs=True)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            disc_loss = disc_loss.mean() if disc_loss is not None else None
            gen_loss = gen_loss.mean() if gen_loss is not None else None
            retrieval_loss = (
                retrieval_loss.mean() if retrieval_loss is not None else None
            )
            gen_retrieval_loss = (
                gen_retrieval_loss.mean() if gen_retrieval_loss is not None else None
            )
            gen_mlm_loss = gen_mlm_loss.mean() if gen_mlm_loss is not None else None
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            gen_loss = (
                gen_loss / self.args.gradient_accumulation_steps
                if gen_loss is not None
                else None
            )
            disc_loss = (
                disc_loss / self.args.gradient_accumulation_steps
                if disc_loss is not None
                else None
            )
            retrieval_loss = (
                retrieval_loss / self.args.gradient_accumulation_steps
                if retrieval_loss is not None
                else None
            )
            gen_retrieval_loss = (
                gen_retrieval_loss / self.args.gradient_accumulation_steps
                if gen_retrieval_loss is not None
                else None
            )
            gen_mlm_loss = (
                gen_mlm_loss / self.args.gradient_accumulation_steps
                if gen_mlm_loss is not None
                else None
            )
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        disc_probs = None
        disc_pred_labels = 0
        disc_accuracy = 0
        disc_f1 = 0
        disc_precision = 0
        disc_auc = 0

        # calculate discriminator metrics
        # calculate disc. metrics
        if disc_logits is not None:
            disc_logits = disc_logits.detach()
            disc_probs = torch.sigmoid(disc_logits).detach().view(-1)
            disc_pred_labels = (disc_probs > 0.5).long()
            discriminator_labels  = discriminator_labels.reshape(-1).long().detach()
            disc_pred_labels  = disc_pred_labels.view(-1).detach()
            # discriminator_labels_np = discriminator_labels.view(-1).cpu().numpy()
            # disc_pred_labels_np = disc_pred_labels.view(-1).cpu().numpy()
            disc_accuracy = torch.mean(
                (disc_pred_labels == discriminator_labels).float()).detach()
            # disc_f1 = torch.as_tensor(
            #     f1_score(discriminator_labels_np, disc_pred_labels_np)
            # )
            # disc_precision = torch.as_tensor(
            #     precision_score(discriminator_labels_np, disc_pred_labels_np)
            # )
            # disc_auc = torch.as_tensor(
            #     roc_auc_score(discriminator_labels_np, disc_pred_labels_np)
            # )
            disc_f1 = torchmetrics.functional.f1(disc_pred_labels, discriminator_labels, multiclass=False).detach()
            disc_precision = torchmetrics.functional.precision(disc_pred_labels, discriminator_labels, multiclass=False).detach()
            disc_auc = torch.zeros_like(disc_precision).detach()
        disc_loss = disc_loss.detach() if disc_loss is not None else None
        gen_loss = gen_loss.detach() if gen_loss is not None else None
        retrieval_loss = retrieval_loss.detach() if retrieval_loss is not None else None
        gen_retrieval_loss = gen_retrieval_loss.detach() if gen_retrieval_loss is not None else None
        gen_mlm_loss = gen_mlm_loss.detach() if gen_mlm_loss is not None else None
        return (
            loss.detach(),
            disc_loss,
            gen_loss,
            retrieval_loss,
            gen_retrieval_loss,
            gen_mlm_loss,
            disc_accuracy,
            disc_f1,
            disc_precision,
            disc_auc,
            head_weights,
            head_accuracies,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(
                f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}."
            )
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
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
            state_dict = torch.load(
                os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu"
            )
            self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        logger.info("loading train dataset")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        logger.info("loaded train dataset")
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = (
                len(train_dataloader) // self.args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = (
                    self.args.max_steps // num_update_steps_per_epoch
                    + int(self.args.max_steps % num_update_steps_per_epoch > 0)
                )
            else:
                max_steps = math.ceil(
                    self.args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        delay_optimizer_creation = (
            self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        )
        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * world_size
        )
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, "trainer_state.json")
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = (
            self.hp_name(trial) if self.hp_name is not None else None
        )
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        tr_disc_loss = torch.tensor(0.0).to(self.args.device)
        tr_gen_loss = torch.tensor(0.0).to(self.args.device)
        tr_disc_f1 = torch.tensor(0.0).to(self.args.device)
        tr_gen_retrieval_loss = torch.tensor(0.0).to(self.args.device)
        tr_gen_mlm_loss = torch.tensor(0.0).to(self.args.device)
        tr_disc_precision = torch.tensor(0.0).to(self.args.device)
        tr_disc_accuracy = torch.tensor(0.0).to(self.args.device)
        tr_disc_auc = torch.tensor(0.0).to(self.args.device)
        tr_retrieval_loss = torch.tensor(0.0).to(self.args.device)
        tr_head_weights = torch.zeros(self.num_instances).to(self.args.device)
        tr_head_accuracies = torch.zeros(self.num_instances).to(self.args.device)

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            self.args,
            self.state,
            self.control,
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(
        #         wait=2,
        #         warmup=2,
        #         active=6,
        #         repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.args.output_dir),
        #     with_stack=True
        # ) as profiler:
        if True:
            for epoch in range(epochs_trained, num_train_epochs):
                if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader.sampler, DistributedSampler
                ):
                    train_dataloader.sampler.set_epoch(epoch)

                if is_torch_tpu_available():
                    parallel_loader = pl.ParallelLoader(
                        train_dataloader, [self.args.device]
                    ).per_device_loader(self.args.device)
                    epoch_iterator = parallel_loader
                else:
                    epoch_iterator = train_dataloader

                # Reset the past mems state at the beginning of each epoch if necessary.
                if self.args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(epoch_iterator)
                    if train_dataset_is_sized
                    else self.args.max_steps * self.args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(
                    self.args, self.state, self.control
                )
                for step, inputs in enumerate(epoch_iterator):

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(
                            self.args, self.state, self.control
                        )

                    if (
                        ((step + 1) % self.args.gradient_accumulation_steps != 0)
                        and self.args.local_rank != -1
                        and self.args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            (
                                cur_tr_loss,
                                cur_disc_loss,
                                cur_gen_loss,
                                cur_retrieval_loss,
                                cur_gen_retrieval_loss,
                                cur_gen_mlm_loss,
                                cur_disc_accuracy,
                                cur_disc_f1,
                                cur_disc_precision,
                                cur_disc_auc,
                                cur_head_weights,
                                cur_head_accuracies,
                            ) = self.training_step(model, inputs)
                            tr_loss += cur_tr_loss
                            if cur_disc_loss is not None:
                                tr_disc_loss += cur_disc_loss
                            if cur_gen_loss is not None:
                                tr_gen_loss += cur_gen_loss
                            if cur_disc_accuracy is not None:
                                tr_disc_accuracy += cur_disc_accuracy
                            if cur_disc_f1 is not None:
                                tr_disc_f1 += cur_disc_f1
                            if cur_disc_precision is not None:
                                tr_disc_precision += cur_disc_precision
                            if cur_disc_auc is not None:
                                tr_disc_auc += cur_disc_auc
                            if cur_retrieval_loss is not None:
                                tr_retrieval_loss += cur_retrieval_loss
                            if cur_gen_retrieval_loss is not None:
                                tr_gen_retrieval_loss += cur_gen_retrieval_loss
                            if cur_gen_mlm_loss is not None:
                                tr_gen_mlm_loss += cur_gen_mlm_loss
                            if cur_head_weights is not None:
                                tr_head_weights += cur_head_weights
                            if cur_head_accuracies is not None:
                                tr_head_accuracies += cur_head_accuracies
                    else:
                        (
                            cur_tr_loss,
                            cur_disc_loss,
                            cur_gen_loss,
                            cur_retrieval_loss,
                            cur_gen_retrieval_loss,
                            cur_gen_mlm_loss,
                            cur_disc_accuracy,
                            cur_disc_f1,
                            cur_disc_precision,
                            cur_disc_auc,
                            cur_head_weights,
                            cur_head_accuracies,
                        ) = self.training_step(model, inputs)
                        tr_loss += cur_tr_loss
                        if cur_disc_loss is not None:
                            tr_disc_loss += cur_disc_loss
                        if cur_gen_loss is not None:
                            tr_gen_loss += cur_gen_loss
                        if cur_disc_accuracy is not None:
                            tr_disc_accuracy += cur_disc_accuracy
                        if cur_disc_f1 is not None:
                            tr_disc_f1 += cur_disc_f1
                        if cur_disc_precision is not None:
                            tr_disc_precision += cur_disc_precision
                        if cur_disc_auc is not None:
                            tr_disc_auc += cur_disc_auc
                        if cur_retrieval_loss is not None:
                            tr_retrieval_loss += cur_retrieval_loss
                        if cur_gen_retrieval_loss is not None:
                            tr_gen_retrieval_loss += cur_gen_retrieval_loss
                        if cur_gen_mlm_loss is not None:
                            tr_gen_mlm_loss += cur_gen_mlm_loss
                        if cur_head_weights is not None:
                            tr_head_weights += cur_head_weights
                        if cur_head_accuracies is not None:
                            tr_head_accuracies += cur_head_accuracies

                    self._total_flos += float(self.floating_point_ops(inputs))

                    # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    if self.deepspeed:
                        self.deepspeed.step()

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= self.args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # Gradient clipping
                        if (
                            self.args.max_grad_norm is not None
                            and self.args.max_grad_norm > 0
                            and not self.deepspeed
                        ):
                            # deepspeed does its own clipping

                            if self.use_amp:
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(self.args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer)
                                    if self.use_apex
                                    else model.parameters(),
                                    self.args.max_grad_norm,
                                )

                        # Optimizer step
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            xm.optimizer_step(self.optimizer)
                        elif self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        if not self.deepspeed:
                            self.lr_scheduler.step()
                        # profiler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(
                            self.args, self.state, self.control
                        )

                        self._maybe_log_save_evaluate(
                            tr_loss,
                            tr_disc_loss,
                            tr_gen_loss,
                            tr_retrieval_loss,
                            tr_gen_retrieval_loss,
                            tr_gen_mlm_loss,
                            tr_disc_accuracy,
                            tr_disc_f1,
                            tr_disc_precision,
                            tr_disc_auc,
                            tr_head_weights,
                            tr_head_accuracies,
                            model,
                            trial,
                            epoch,
                        )

                    if (
                        self.control.should_epoch_stop
                        or self.control.should_training_stop
                    ):
                        break

                self.control = self.callback_handler.on_epoch_end(
                    self.args, self.state, self.control
                )

                self._maybe_log_save_evaluate(
                    tr_loss,
                    tr_disc_loss,
                    tr_gen_loss,
                    tr_retrieval_loss,
                    tr_gen_retrieval_loss,
                    tr_gen_mlm_loss,
                    tr_disc_accuracy,
                    tr_disc_f1,
                    tr_disc_precision,
                    tr_disc_auc,
                    tr_head_weights,
                    tr_head_accuracies,
                    model,
                    trial,
                    epoch,
                )

                if self.args.tpu_metrics_debug or self.args.debug:
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if (
            self.args.load_best_model_at_end
            and self.state.best_model_checkpoint is not None
        ):
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(
                    self.state.best_model_checkpoint
                )
                if self.place_model_on_device:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(
                    os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
                )
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        if self.deepspeed:
            # free up any memory that might be useful for eval
            self.deepspeed = None
            self.optimizer = None
            self.lr_scheduler = None
            self.model_wrapped = self.model
            gc.collect()  # force memory release
            # to restore normal behavior outside of train replay the place_model_on_device logic w/o deepspeed
            self.place_model_on_device = self.args.place_model_on_device
            if self.is_model_parallel:
                self.place_model_on_device = False

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        return TrainOutput(
            self.state.global_step,
            self._total_loss_scalar / self.state.global_step,
            metrics,
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                # save generator and discriminator models
                if self.model.discriminator is not None:
                    discriminator_state_dict = self.model.discriminator.state_dict()
                    unwrap_model(self.model.discriminator).save_pretrained(
                        os.path.join(output_dir, "discriminator"),
                        state_dict=discriminator_state_dict,
                    )

                if self.model.generator is not None:
                    generator_state_dict = self.model.generator.state_dict()
                    unwrap_model(self.model.generator).save_pretrained(
                        os.path.join(output_dir, "generator"),
                        state_dict=generator_state_dict,
                    )

        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
