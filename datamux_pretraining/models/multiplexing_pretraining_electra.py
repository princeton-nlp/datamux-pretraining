from dis import dis
from multiprocessing import reduction
import numpy as np
import torch
from torch import nn
from torch import import_ir_module
import torch.nn.functional as F
from transformers import (
    DataCollatorForLanguageModeling,
)
from dataclasses import dataclass
import tracemalloc
from transformers.utils import logging
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.models.electra.modeling_electra import (
    ElectraDiscriminatorPredictions,
    ElectraForPreTrainingOutput,
    ElectraClassificationHead,
    ElectraGeneratorPredictions,
    ElectraLayer,
)
import math
from transformers.activations import gelu
import time
from datamux_pretraining.models.utils import (
    random_encoding,
    binary_encoding,
)
from datamux_pretraining.models.multiplexing_pretraining_utils import (
    SequenceClassifierOutputMuxed,
    RetrievalHeadIndexDemultiplexing,
    RetrievalHeadIndexPosDemultiplexing,
    RetrievalHeadIndexPosDemultiplexingBERT,
    IndexDemultiplexerTokenLevel,
    IndexPosDemultiplexerTokenLevel,
    IndexPosDemuxModule,
)
from scipy.stats import ortho_group, special_ortho_group
from datamux_pretraining.models.utils import gen_attn_mask

logger = logging.get_logger(__name__)


class ELECTRAModel(nn.Module):
    def __init__(
        self,
        discriminator,
        discriminator_config,
        tokenizer,
        generator=None,
        generator_config=None,
        loss_weights=(1.0, 50.0),
    ):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        # self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.0, 1.0)
        self.tokenizer = tokenizer
        self.discriminator_config = discriminator_config
        self.generator_config = generator_config
        self.gen_loss_fc = nn.CrossEntropyLoss()
        self.disc_loss_fc = nn.BCEWithLogitsLoss()
        self.loss_weights = loss_weights

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, return_dict=None
    ):
        mask_token = 103
        is_mlm_applied = input_ids == mask_token
        mlm_gen_logits = None
        gen_loss = None
        gen_retrieval_loss = None
        gen_mlm_loss = None
        if self.generator is not None:
            gen_outputs = self.generator(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            # (B, L, vocab size)
            # reduce size to save space and speed
            gen_logits = gen_outputs["logits"]
            mlm_gen_logits = gen_logits[
                is_mlm_applied, :
            ]  # ( #mlm_positions, vocab_size)
            # loss calculation
            gen_loss = gen_outputs["loss"]
            gen_retrieval_loss = None
            gen_mlm_loss = None
            if "retrieval_loss" in gen_outputs:
                gen_retrieval_loss = gen_outputs["retrieval_loss"]
            if "task_loss" in gen_outputs:
                gen_mlm_loss = gen_outputs["task_loss"]
            with torch.no_grad():
                # sampling
                pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )
                # produce inputs for discriminator
                generated = input_ids.clone()  # (B,L)
                generated[is_mlm_applied] = pred_toks  # (B,L)
                # produce labels for discriminator
                is_replaced = is_mlm_applied.clone()  # (B,L)
                is_replaced[is_mlm_applied] = (
                    pred_toks != labels[is_mlm_applied]
                )  # (B,L)
        else:
            # replace the masked tokens with other random tokens from the vocab
            generated = input_ids.clone()  # (B,L)
            rand_toks = torch.randint_like(
                generated, self.discriminator_config.vocab_size
            )
            generated[is_mlm_applied] = rand_toks[is_mlm_applied]  # (B,L)
            is_replaced = is_mlm_applied.clone()  # (B,L)
        # pass the generated tokens to the discriminator only if loss weight is non-zero
        discriminator_hidden_states = None
        if self.loss_weights[1] > 0:
            # s_t = time.time()
            outputs = self.discriminator(generated, attention_mask, return_dict=True)
            disc_logits = outputs["logits"]
            discriminator_hidden_states = (
                outputs["hidden_states"] if "hidden_states" in outputs else None
            )
            retrieval_loss = (
                None if "retrieval_loss" not in outputs else outputs["retrieval_loss"]
            )
            if self.discriminator_config.demuxing_variant == "index":
                # TODO: !!!! this needs to be cleaned up, could lead to bugs
                attention_mask = attention_mask[
                    :, : -(self.discriminator_config.num_instances + 1)
                ]
                is_replaced = is_replaced[
                    :, : -(self.discriminator_config.num_instances + 1)
                ]
            disc_logits_flat = disc_logits.masked_select(
                attention_mask.bool()
            )  # -> 1d tensor
            is_replaced_flat = is_replaced.masked_select(attention_mask.bool())  # -> 1d
            disc_loss = self.disc_loss_fc(
                disc_logits_flat.float(), is_replaced_flat.float()
            )
            # add retrieval loss here
            disc_retrieval_loss = (
                self.discriminator_config.retrieval_loss_coeff * retrieval_loss
                + self.discriminator_config.task_loss_coeff * disc_loss
                if retrieval_loss is not None
                else disc_loss
            )
        else:
            disc_logits = None
            retrieval_loss = None
            disc_logits = None
            disc_loss = None
            disc_retrieval_loss = 0

        loss = (
            gen_loss * self.loss_weights[0] + disc_retrieval_loss * self.loss_weights[1]
            if gen_loss is not None
            else disc_retrieval_loss * self.loss_weights[1]
        )

        if not return_dict:
            return (
                loss,
                disc_loss,
                gen_loss,
                retrieval_loss,
                gen_retrieval_loss,
                gen_mlm_loss,
                disc_logits,
                None,  # mlm_gen_logits,
                generated,
                is_replaced,
            )
        return ElectraOutput(
            loss=loss,
            disc_loss=disc_loss,
            gen_loss=gen_loss,
            retrieval_loss=retrieval_loss,
            gen_retrieval_loss=gen_retrieval_loss,
            gen_mlm_loss=gen_mlm_loss,
            disc_logits=disc_logits,
            sampled_input_ids=generated,
            corruption_applied=is_replaced,
            hidden_states=discriminator_hidden_states,
        )

    def sample(self, logits):
        return F.gumbel_softmax(logits).argmax(dim=-1)


@dataclass
class ElectraOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    disc_loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
    gen_retrieval_loss: Optional[torch.FloatTensor] = None
    gen_mlm_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    disc_logits: torch.FloatTensor = None
    mlm_gen_logits: torch.FloatTensor = None
    sampled_input_ids: torch.LongTensor = None
    corruption_applied: torch.BoolTensor = None


class MuxedElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.electra = ElectraModel(config)

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(config)
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        self.init_weights()

        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

        # multiplexing initialization

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=config.hidden_size)
            U_ = special_ortho_group.rvs(dim=config.hidden_size)
            for i in range(self.num_instances):
                G_ = np.zeros((config.hidden_size, config.hidden_size))
                l = i * (config.hidden_size // self.num_instances)
                r = l + (config.hidden_size // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        elif self.muxing_variant == "gaussian_attention":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
            self.muxing_attention = ElectraLayer(config)
            self.cross_instances_linear = nn.Linear(config.embedding_size, d_model)
            self.cross_instances_layernorm = nn.LayerNorm(d_model)

        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2) + 680
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids[:, : -(num_instances + 1)]], dim=1)
            modified_seq_length = seq_length
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids[:, 0:1] = cls_tokens
            modified_seq_length = seq_length
            special_tokens_end_position = 0

        else:
            raise NotImplementedError()
        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        elif self.muxing_variant == "gaussian_attention":
            embedding_output_intermediate = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_intermediate * instance_embed.unsqueeze(0)
            )

            embedding_output_cross_instance = torch.mean(
                embedding_output_intermediate, dim=1
            )
            embedding_output_cross_instance = self.cross_instances_linear(
                embedding_output_cross_instance
            )
            embedding_output_cross_instance = gelu(embedding_output_cross_instance)
            embedding_output_cross_instance = self.cross_instances_layernorm(
                embedding_output_cross_instance
            )

            embedding_output_intermediate = embedding_output_intermediate.view(
                modified_batch_size * num_instances, modified_seq_length, embedding_dim
            )
            # pass throughh attention layer
            embedding_output_attention = self.muxing_attention(
                embedding_output_intermediate
            )
            embedding_output_attention = embedding_output_attention[0]
            embedding_output_cross_instance = embedding_output_cross_instance.unsqueeze(
                1
            ).expand(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            # average across the instances, and add the cross instance attention
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_attention = (
                embedding_output_attention + embedding_output_cross_instance
            )
            embedding_output = torch.mean(embedding_output_attention, dim=1)
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        demuxed_sequence_output = self.demultiplexer(sequence_output)
        logits = self.discriminator_predictions(demuxed_sequence_output)
        # retrieval loss calculation
        instance_labels = torch.full(
            (modified_batch_size, modified_seq_length),
            0,
            device=input_ids.device,
        ).long()
        # skip the cls and prefix tokens
        instance_labels[:, special_tokens_end_position:] = torch.randint(
            num_instances,
            (modified_batch_size, modified_seq_length - special_tokens_end_position),
            device=input_ids.device,
        )

        # index into input ids to get the corresponding labels
        input_ids = input_ids.view(modified_batch_size, num_instances, -1)
        input_ids = input_ids.permute(0, 2, 1)

        retrieval_labels = input_ids[
            torch.arange(modified_batch_size, device=input_ids.device)
            .unsqueeze(1)
            .expand(modified_batch_size, modified_seq_length),
            torch.arange(modified_seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(modified_batch_size, modified_seq_length),
            instance_labels,
        ]
        retrieval_labels[:, :special_tokens_end_position] = -100

        retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)

        retrieval_loss = None
        task_loss = None
        loss = None

        retrieval_loss_fct = nn.CrossEntropyLoss()

        retrieval_loss = retrieval_loss_fct(
            retrieval_predictions.view(-1, self.config.vocab_size),
            retrieval_labels.view(-1),
        )

        if labels is not None:
            labels = labels[: (modified_batch_size * num_instances)]

            loss_fct = nn.BCEWithLogitsLoss()

            if attention_mask is not None:
                loss_fct = nn.BCEWithLogitsLoss()

                active_loss = (
                    attention_mask.view(-1, demuxed_sequence_output.shape[1]) == 1
                )
                active_logits = logits.view(-1, demuxed_sequence_output.shape[1])[
                    active_loss
                ]
                active_labels = labels[active_loss]

                task_loss = loss_fct(active_logits, active_labels.float())
                loss = (self.task_loss_coeff * task_loss) + (
                    self.retrieval_loss_coeff * retrieval_loss
                )

        # make logits align with the length of the input sequence
        logits = logits[:, special_tokens_end_position:]
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MuxedElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
            hidden_states=demuxed_sequence_output,
        )


class MuxedElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(config)
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        self.init_weights()

        self.classifier = ElectraClassificationHead(config)

        # multiplexing initialization

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=config.hidden_size)
            U_ = special_ortho_group.rvs(dim=config.hidden_size)
            for i in range(self.num_instances):
                G_ = np.zeros((config.hidden_size, config.hidden_size))
                l = i * (config.hidden_size // self.num_instances)
                r = l + (config.hidden_size // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

        self.head_accuracies = None
        self.head_acc_smooth_coeff = 0.8
        self.head_acc_gamma = 10
        # Initialize weights and apply final processing
        # self.post_init()
        # self.anchor = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dummy_samples_added = 0
        if input_ids.shape[0] % self.num_instances != 0:
            dummy_samples_added = self.num_instances - (
                input_ids.shape[0] % self.num_instances
            )
            pad_input_ids = input_ids[
                torch.randint(0, input_ids.shape[0], (dummy_samples_added,))
            ]
            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
            pad_attention_mask = attention_mask[
                torch.randint(0, attention_mask.shape[0], (dummy_samples_added,))
            ]
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2) + 680
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            # input_ids = torch.cat([cls_tokens, input_ids], dim=1)
            # modified_seq_length = seq_length + 1
            modified_seq_length = seq_length
            special_tokens_end_position = 1

        else:
            raise NotImplementedError()

        # if self.anchor is None:
        #     self.anchor = input_ids[0:1]

        # anchor_mask = (input_ids == self.anchor).all(dim=1)
        # if anchor_mask.sum() > 0:
        #     logger.warning("Anchor token found in input_ids")
        #     logger.warning(f"Anchor id: {torch.where(anchor_mask)[0]}")
        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding
        if self.demuxing_variant != "index":
            demuxed_sequence_output = self.demultiplexer(sequence_output[:, 0:1, :])
        else:
            demuxed_sequence_output = self.demultiplexer(sequence_output)
            demuxed_sequence_output = demuxed_sequence_output[
                :, num_instances : num_instances + 1, :
            ]
        logits = self.classifier(demuxed_sequence_output)

        if labels is not None:
            # retrieval loss calculation
            instance_labels = torch.full(
                (modified_batch_size, modified_seq_length),
                0,
                device=input_ids.device,
            ).long()
            # skip the cls and prefix tokens
            instance_labels[:, special_tokens_end_position:] = torch.randint(
                num_instances,
                (
                    modified_batch_size,
                    modified_seq_length - special_tokens_end_position,
                ),
                device=input_ids.device,
            )

            # index into input ids to get the corresponding labels
            input_ids = input_ids.view(modified_batch_size, num_instances, -1)
            input_ids = input_ids.permute(0, 2, 1)

            retrieval_labels = input_ids[
                torch.arange(modified_batch_size, device=input_ids.device)
                .unsqueeze(1)
                .expand(modified_batch_size, modified_seq_length),
                torch.arange(modified_seq_length, device=input_ids.device)
                .unsqueeze(0)
                .expand(modified_batch_size, modified_seq_length),
                instance_labels,
            ]
            retrieval_labels[:, :special_tokens_end_position] = -100

            pad_mask = retrieval_labels == 0
            # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
            pad_mask_wipe = pad_mask
            non_pad_mask_wipe = (
                ~pad_mask
                & torch.bernoulli(
                    torch.full(
                        retrieval_labels.shape,
                        1 - self.config.retrieval_percentage,
                        device=input_ids.device,
                    )
                ).bool()
            )
            retrieval_labels[non_pad_mask_wipe] = -100

            retrieval_labels[pad_mask_wipe] = -100

            retrieval_predictions = self.retrieval_head(
                sequence_output, instance_labels
            )
        retrieval_loss = None
        task_loss = None
        loss = None
        if dummy_samples_added != 0:
            logits = logits[:-dummy_samples_added]
        if labels is not None:
            # labels = labels[: (modified_batch_size * num_instances)]
            assert len(labels.shape) == 1  # assert one dimension
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                task_loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                task_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_fct = nn.CrossEntropyLoss()
            retrieval_loss = loss_fct(
                retrieval_predictions.view(-1, self.config.vocab_size),
                retrieval_labels.view(-1),
            )
            loss = (self.task_loss_coeff * task_loss) + (
                self.retrieval_loss_coeff * retrieval_loss
            )
        # make logits align with the length of the input sequence
        # logits = logits[:, special_tokens_end_position:]
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputMuxed(
            loss=loss,
            logits=logits,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
            retrieval_predictions=None,
            retrieval_instance_labels=None,
            hidden_states=demuxed_sequence_output,
        )


class MuxedElectraForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(config)
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        self.init_weights()
        classifier_dropout = config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # multiplexing initialization

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=config.hidden_size)
            U_ = special_ortho_group.rvs(dim=config.hidden_size)
            for i in range(self.num_instances):
                G_ = np.zeros((config.hidden_size, config.hidden_size))
                l = i * (config.hidden_size // self.num_instances)
                r = l + (config.hidden_size // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)
        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dummy_samples_added = 0
        if input_ids.shape[0] % self.num_instances != 0:
            dummy_samples_added = self.num_instances - (
                input_ids.shape[0] % self.num_instances
            )
            pad_input_ids = input_ids[
                torch.randint(0, input_ids.shape[0], (dummy_samples_added,))
            ]
            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
            pad_attention_mask = attention_mask[
                torch.randint(0, attention_mask.shape[0], (dummy_samples_added,))
            ]
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2) + 680
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            # input_ids = torch.cat([cls_tokens, input_ids], dim=1)
            # modified_seq_length = seq_length + 1
            modified_seq_length = seq_length
            special_tokens_end_position = 0

        else:
            raise NotImplementedError()

        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)

        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        demuxed_sequence_output = self.demultiplexer(sequence_output)
        demuxed_sequence_output = self.dropout(demuxed_sequence_output)
        logits = self.classifier(demuxed_sequence_output)

        if labels is not None:
            # retrieval loss calculation
            instance_labels = torch.full(
                (modified_batch_size, modified_seq_length),
                0,
                device=input_ids.device,
            ).long()
            # skip the cls and prefix tokens
            instance_labels[:, special_tokens_end_position:] = torch.randint(
                num_instances,
                (
                    modified_batch_size,
                    modified_seq_length - special_tokens_end_position,
                ),
                device=input_ids.device,
            )

            # index into input ids to get the corresponding labels
            input_ids = input_ids.view(modified_batch_size, num_instances, -1)
            input_ids = input_ids.permute(0, 2, 1)

            retrieval_labels = input_ids[
                torch.arange(modified_batch_size, device=input_ids.device)
                .unsqueeze(1)
                .expand(modified_batch_size, modified_seq_length),
                torch.arange(modified_seq_length, device=input_ids.device)
                .unsqueeze(0)
                .expand(modified_batch_size, modified_seq_length),
                instance_labels,
            ]
            retrieval_labels[:, :special_tokens_end_position] = -100

            pad_mask = retrieval_labels == 0
            # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
            pad_mask_wipe = pad_mask
            non_pad_mask_wipe = (
                ~pad_mask
                & torch.bernoulli(
                    torch.full(
                        retrieval_labels.shape,
                        1 - self.config.retrieval_percentage,
                        device=input_ids.device,
                    )
                ).bool()
            )
            retrieval_labels[non_pad_mask_wipe] = -100

            retrieval_labels[pad_mask_wipe] = -100

            retrieval_predictions = self.retrieval_head(
                sequence_output, instance_labels
            )

        retrieval_loss = None
        task_loss = None
        loss = None

        if dummy_samples_added != 0:
            logits = logits[:-dummy_samples_added]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = logits[:, special_tokens_end_position:, :]
            task_loss = loss_fct(logits.reshape(-1, self.num_labels), labels.view(-1))
            retrieval_loss_fct = nn.CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(
                retrieval_predictions.view(-1, self.config.vocab_size),
                retrieval_labels.view(-1),
            )
            loss = (self.task_loss_coeff * task_loss) + (
                self.retrieval_loss_coeff * retrieval_loss
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputMuxed(
            loss=loss,
            logits=logits,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
        )


class MuxedElectraForMaskedLM(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(config)
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexing(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        d_model = config.embedding_size
        instance_embedding = None

        if self.muxing_variant == "gaussian_hadamard":
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
            )
        elif self.muxing_variant == "random_ortho":
            instance_embedding = [
                torch.from_numpy(ortho_group.rvs(config.hidden_size)).float()
                for _ in range(self.num_instances)
            ]
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "random_ortho_low_rank":
            # create low rank matrix
            instance_embedding = []
            H_ = special_ortho_group.rvs(dim=d_model)
            U_ = special_ortho_group.rvs(dim=d_model)
            for i in range(self.num_instances):
                G_ = np.zeros((d_model, d_model))
                l = i * (d_model // self.num_instances)
                r = l + (d_model // self.num_instances)
                H_copy = H_.copy()
                G_[:, l:r] = H_copy[:, l:r]
                G_ = U_ @ G_.T
                instance_embedding.append(torch.from_numpy(G_).float())
            instance_embedding = torch.stack(instance_embedding, dim=0)

        elif self.muxing_variant == "binary_hadamard":
            instance_embedding = binary_encoding(
                self.num_instances, d_model, epsilon=config.binary_hadamard_epsilon
            )
        else:
            raise NotImplementedError()

        if instance_embedding is not None:
            self.instance_embedding = torch.nn.Parameter(instance_embedding)
        else:
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=self.gaussian_hadamard_norm
            )

        if not config.learn_muxing:
            self.instance_embedding.requires_grad = False
        else:
            self.instance_embedding.requires_grad = True

        self.electra = ElectraModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):
        # s_t = time.time()
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # get input embeddings and average over N instances
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        num_instances = self.num_instances
        past_key_values_length = 0

        modified_batch_size = batch_size // num_instances
        modified_seq_length = None
        special_tokens_end_position = None
        if self.demuxing_variant == "index":

            # add the prefix
            # [CLS1, <s>, <s>, <s>, <s>]
            # [<s>, CLS2, <s>, <s>, <s>]
            # [<s>, <s>, CLS3, <s>, <s>]
            # [<s>, <s>, <s>, CLS4, <s>]
            # [<s>, <s>, <s>, <s>, CLS5]
            # let us just assume the last 5 tokens barring the masked token
            # are the cls tokens (easiest way to make use of existing vocab)

            # prefix 5 x 5

            prefix = torch.full(
                (num_instances, num_instances), 680, device=input_ids.device
            )
            prefix[
                torch.arange(num_instances, device=input_ids.device),
                torch.arange(num_instances, device=input_ids.device),
            ] = (
                -(torch.arange(num_instances, device=input_ids.device) + 2) + 680
            )

            # [-2   <s>, <s>, <s>, <s>]
            # [<s>, -3, <s>, <s>, <s>]
            # [<s>, <s>, -4, <s>, <s>]
            # [<s>, <s>, <s>, -5, <s>]
            # [<s>, <s>, <s>, <s>, -6]
            # +  size of vocab
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            prefix = torch.cat([prefix, cls_tokens], dim=1)

            prefix = prefix.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids = torch.cat([prefix, input_ids], dim=1)
            modified_seq_length = seq_length + num_instances + 1
            special_tokens_end_position = num_instances + 1

        elif self.demuxing_variant == "index_pos":
            cls_tokens = torch.full((num_instances, 1), 101, device=input_ids.device)
            cls_tokens = cls_tokens.repeat(modified_batch_size, 1)
            input_ids = input_ids[: (modified_batch_size * num_instances)]
            input_ids[:, 0:1] = cls_tokens
            modified_seq_length = seq_length
            special_tokens_end_position = 0

        else:
            raise NotImplementedError()
        # concatenate
        embedding_output = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        _, _, embedding_dim = embedding_output.shape
        if (
            self.muxing_variant == "random_ortho"
            or self.muxing_variant == "random_ortho_low_rank"
        ):
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            embedding_output = torch.matmul(
                self.instance_embedding, embedding_output.permute(0, 1, 3, 2)
            )
            # swap the last 2 dimensions again
            embedding_output = embedding_output.permute(0, 1, 3, 2)
            # average across the instances
            embedding_output = torch.sum(embedding_output, dim=1) / math.sqrt(
                self.num_instances
            )
        else:
            embedding_output = embedding_output.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            # extract relevant instance embeddings
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output = embedding_output * instance_embed.unsqueeze(0)

            embedding_output = torch.mean(embedding_output, dim=1)
        # logger.warn(f"Time taken to embedding: {time.time() - s_t}")
        # s_t = time.time()
        outputs = self.electra(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )
        # logger.warn(f"Time taken to forward: {time.time() - s_t}")
        sequence_output = outputs[0]
        # fancy indexing to get the instance position embedding

        # retrieval loss calculation
        instance_labels = torch.full(
            (modified_batch_size, modified_seq_length),
            0,
            device=input_ids.device,
        ).long()
        # skip the cls and prefix tokens
        instance_labels[:, special_tokens_end_position:] = torch.randint(
            num_instances,
            (modified_batch_size, modified_seq_length - special_tokens_end_position),
            device=input_ids.device,
        )

        # index into input ids to get the corresponding labels
        input_ids = input_ids.view(modified_batch_size, num_instances, -1)
        input_ids = input_ids.permute(0, 2, 1)
        # s_t = time.time()
        retrieval_labels = input_ids[
            torch.arange(modified_batch_size, device=input_ids.device)
            .unsqueeze(1)
            .expand(modified_batch_size, modified_seq_length),
            torch.arange(modified_seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(modified_batch_size, modified_seq_length),
            instance_labels,
        ]
        retrieval_labels[:, :special_tokens_end_position] = -100

        pad_mask = retrieval_labels == 0
        # wipe of 1 - (0.1  *  retrieval percentage) of pad tokens
        pad_mask_wipe = pad_mask
        non_pad_mask_wipe = (
            ~pad_mask
            & torch.bernoulli(
                torch.full(
                    retrieval_labels.shape,
                    1 - self.config.retrieval_percentage,
                    device=input_ids.device,
                )
            ).bool()
        )
        retrieval_labels[non_pad_mask_wipe] = -100

        retrieval_labels[pad_mask_wipe] = -100

        retrieval_predictions = self.retrieval_head(sequence_output, instance_labels)
        # only run the expensive head on masked tokens
        is_mlm_applied = labels != -100
        # s_t = time.time()
        mlm_logits = self.retrieval_head(sequence_output, None, is_mlm_applied, labels)
        logits = torch.zeros(
            batch_size,
            modified_seq_length,
            self.config.vocab_size,
            dtype=mlm_logits.dtype,
            device=mlm_logits.device,
        )
        logits[is_mlm_applied] = mlm_logits
        retrieval_loss = None
        task_loss = None
        loss = None

        retrieval_loss_fct = nn.CrossEntropyLoss()

        retrieval_loss = retrieval_loss_fct(
            retrieval_predictions.view(-1, self.config.vocab_size),
            retrieval_labels.view(-1),
        )
        if labels is not None:
            labels = labels[: (modified_batch_size * num_instances)]

            if attention_mask is not None:
                loss_fct = nn.CrossEntropyLoss()

                active_loss = attention_mask.view(-1, modified_seq_length) == 1
                active_logits = logits[active_loss]
                active_labels = labels[active_loss]

                task_loss = loss_fct(active_logits, active_labels)
                loss = (self.task_loss_coeff * task_loss) + (
                    self.retrieval_loss_coeff * retrieval_loss
                )

        # make logits align with the length of the input sequence
        logits = logits[:, special_tokens_end_position:]
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MuxedElectraForPreTrainingOutput(
            loss=loss, logits=logits, task_loss=task_loss, retrieval_loss=retrieval_loss
        )


@dataclass
class MuxedElectraForPreTrainingOutput(ElectraForPreTrainingOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    retrieval_loss: torch.FloatTensor = None
    task_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
