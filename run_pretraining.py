#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
from re import I
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

from datamux_pretraining.models.mlm_pretraining_trainer import MuxedMLMTrainer
from datamux_pretraining.models.electra_pretraining_trainer import MuxedElectraTrainer
from datamux_pretraining.models.multiplexing_pretraining_electra import (
    ELECTRAModel,
    MuxedElectraForMaskedLM,
    MuxedElectraForPreTraining,
)
from datamux_pretraining.models.multiplexing_pretraining_utils import (
    DataCollatorForLanguageModelingMuxed,
    DataCollatorElectra,
)
from datamux_pretraining.models.multiplexing_pretraining_bert import (
    MuxedBertForMaskedLM,
)
from datasets import load_dataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    ElectraForPreTraining,
    ElectraForMaskedLM,
    BertForMaskedLM,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

version_2_modelcls = {
    "electra": MuxedElectraForPreTraining,
    "electra_gen": MuxedElectraForMaskedLM,
    "electra_no_mux": ElectraForPreTraining,
    "electra_no_mux_gen": ElectraForMaskedLM,
    "bert": MuxedBertForMaskedLM,
    "bert_no_mux": BertForMaskedLM,
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    data_dir: Optional[str] = field(
        default="",
        metadata={"help": "Location of datasets for offline loading)."},
    )

    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_pad_tokens: Optional[int] = field(
        default=0, metadata={"help": "max number of pad tokens in pretrained model"}
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    generator_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models for generator in electra models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    generator_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name for generator in electra model"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # multi instance arguments
    num_instances: Optional[int] = field(
        default=5,
        metadata={"help": "Number of instances i.e. N"},
    )
    muxing_variant: Optional[str] = field(
        default="gaussian_hadamard",
        metadata={
            "help": "muxing variant; choose from gaussian_hadamard or random_ortho or binary_hadamard"
        },
    )
    demuxing_variant: Optional[str] = field(
        default="index",
        metadata={"help": "demuxing variant, choose from  'index' or 'mlp'"},
    )
    should_mux: Optional[int] = field(
        default=1,
        metadata={"help": "whether to mux, turn off for non-multiplexed baselines"},
    )
    retrieval_percentage: Optional[float] = field(
        default=1.0,
        metadata={"help": "percentage of tokens to retrieve during inference"},
    )
    retrieval_pretraining: Optional[int] = field(
        default=0,
        metadata={"help": "Retrieval Pretraining"},
    )
    gaussian_hadamard_norm: Optional[float] = field(
        default=1,
        metadata={"help": "Norm of sentence embeddings if we use random projections"},
    )
    binary_hadamard_epsilon: Optional[float] = field(
        default=0,
        metadata={
            "help": "Percentage intersection among binary vectors, default is no intersection"
        },
    )
    retrieval_loss_coeff: Optional[float] = field(
        default=0.1,
        metadata={"help": "Coefficient for retrieval loss"},
    )
    task_loss_coeff: Optional[float] = field(
        default=0.9,
        metadata={"help": "Coefficient for task loss"},
    )
    learn_muxing: Optional[int] = field(
        default=0,
        metadata={"help": "whether instance embeddings are learnt or not"},
    )
    model_version: Optional[str] = field(
        default="electra",
        metadata={
            "help": "pretraining architecture, choose from  'roberta' or 'electra'"
        },
    )
    generator_model_version: Optional[str] = field(
        default="electra",
        metadata={
            "help": "pretraining architecture, choose from  'roberta' or 'electra'"
        },
    )
    gen_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Coefficient for generator loss"},
    )
    disc_loss_weight: Optional[float] = field(
        default=50.0,
        metadata={"help": "Coefficient for discriminator loss"},
    )
    num_hidden_demux_layers: Optional[int] = field(
        default=2,
        metadata={"help": "number of hidden layers for demuxing"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args.to_json_string()}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir
        )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.generator_config_name:
        generator_config = AutoConfig.from_pretrained(
            model_args.generator_config_name, **config_kwargs
        )
    elif model_args.generator_name_or_path:
        generator_config = AutoConfig.from_pretrained(
            model_args.generator_name_or_path, **config_kwargs
        )
    else:
        generator_config = None

    # setting instance params
    config.num_instances = model_args.num_instances
    config.muxing_variant = model_args.muxing_variant
    config.demuxing_variant = model_args.demuxing_variant
    config.retrieval_percentage = model_args.retrieval_percentage
    config.gaussian_hadamard_norm = model_args.gaussian_hadamard_norm
    config.binary_hadamard_epsilon = model_args.binary_hadamard_epsilon
    config.retrieval_loss_coeff = model_args.retrieval_loss_coeff
    config.task_loss_coeff = model_args.task_loss_coeff
    config.learn_muxing = model_args.learn_muxing
    config.num_hidden_demux_layers = model_args.num_hidden_demux_layers
    if "bert" in model_args.model_version:
        config.output_hidden_states = True
        config.output_attentions = True
    if generator_config is not None:
        generator_config.num_instances = model_args.num_instances
        generator_config.muxing_variant = model_args.muxing_variant
        generator_config.demuxing_variant = model_args.demuxing_variant
        generator_config.retrieval_percentage = model_args.retrieval_percentage
        generator_config.gaussian_hadamard_norm = model_args.gaussian_hadamard_norm
        generator_config.binary_hadamard_epsilon = model_args.binary_hadamard_epsilon
        generator_config.retrieval_loss_coeff = model_args.retrieval_loss_coeff
        generator_config.task_loss_coeff = model_args.task_loss_coeff
        generator_config.learn_muxing = model_args.learn_muxing
        generator_config.num_hidden_demux_layers = model_args.num_hidden_demux_layers

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    model_cls = version_2_modelcls[model_args.model_version]
    model_path_supplied = model_args.model_name_or_path is not None
    generator_model_path_supplied = model_args.generator_name_or_path is not None
    # initialize electra slightly differently (2 models creates different settings)
    if "electra" in model_args.model_version:
        # initialize generator model
        if generator_config is not None:
            generator_model_cls = version_2_modelcls[model_args.generator_model_version]
            if generator_model_path_supplied:
                if not issubclass(generator_model_cls, PreTrainedModel):
                    model = model_cls(config=config)
                    state_dict = torch.load(
                        os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
                    )
                    model.load_state_dict(state_dict, strict=False)
                else:
                    generator_model = generator_model_cls.from_pretrained(
                        model_args.generator_name_or_path,
                        config=generator_config,
                    )
            else:
                generator_model = generator_model_cls(config=generator_config)
        else:
            generator_model = None
        # initialize discriminator model
        if model_path_supplied:
            if not issubclass(model_cls, PreTrainedModel):
                # manually load state dict
                model = model_cls(config=config)
                state_dict = torch.load(
                    os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
                )
                model.load_state_dict(state_dict, strict=False)
            else:
                model = model_cls.from_pretrained(
                    model_args.model_name_or_path, config=config
                )
        else:
            model = model_cls(config=config)
        if (
            generator_model is not None
            and generator_config.embedding_size == config.embedding_size
        ):
            model.electra.embeddings = generator_model.electra.embeddings
            # generator_model.generator_lm_head.weight = generator_model.electra.embeddings.word_embeddings.weight        # initialize electra model wrapper
        model = ELECTRAModel(
            model,
            config,
            tokenizer,
            generator=generator_model,
            generator_config=generator_config,
            loss_weights=(model_args.gen_loss_weight, model_args.disc_loss_weight),
        )

    else:
        if model_path_supplied:
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
        else:
            model = model_cls(config=config)

    # model.resize_token_embeddings(len(tokenizer))
    # Preprocessing the datasets.
    # First we tokenize all the texts.

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        # https://github.com/huggingface/transformers/issues/14931

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [
                line
                for line in examples["text"]
                if len(line) > 0 and not line.isspace()
            ]

            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        # https://github.com/huggingface/transformers/issues/14931
        def tokenize_function(examples):
            examples["text"] = [
                line
                for line in examples["text"]
                if line is not None and len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            # load_from_cache_file=False,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]

        if data_args.max_train_samples is not None:
            assert data_args.max_train_samples < 100
            train_samples = int(data_args.max_train_samples / 100 * len(train_dataset))
            train_dataset = train_dataset.select(range(train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            assert data_args.max_eval_samples < 100
            eval_samples = int(data_args.max_eval_samples / 100 * len(eval_dataset))
            eval_dataset = eval_dataset.select(range(eval_samples))
    # for electra, need different collators
    if "electra" not in model_args.model_version:

        train_data_collator = DataCollatorForLanguageModelingMuxed(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            max_pad_tokens=data_args.max_pad_tokens,
        )
        eval_data_collator = DataCollatorForLanguageModelingMuxed(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            max_pad_tokens=data_args.max_pad_tokens,
        )
        trainer = MuxedMLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            train_collator=train_data_collator,
            eval_collator=eval_data_collator,
        )
    else:
        train_data_collator = DataCollatorElectra(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            max_pad_tokens=data_args.max_pad_tokens,
        )

        eval_data_collator = DataCollatorElectra(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            max_pad_tokens=0,
        )
        trainer = MuxedElectraTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            train_collator=train_data_collator,
            eval_collator=eval_data_collator,
            num_instances=model_args.num_instances,
        )
    # Training
    logger.info("starting training")
    if training_args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            int(data_args.max_train_samples / 100 * len(train_dataset))
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        output_dir = training_args.output_dir
        last_checkpoint = get_last_checkpoint(output_dir)

        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        metrics = trainer.evaluate(resume_from_checkpoint=checkpoint)
        max_eval_samples = (
            int(data_args.max_eval_samples / 100 * len(eval_dataset))
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
