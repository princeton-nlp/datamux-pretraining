from dataclasses import dataclass
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
import torch
import numpy as np
from scipy.stats import ortho_group, special_ortho_group
from datamux_pretraining.models.utils import (
    random_encoding,
    binary_encoding,
)
from datamux_pretraining.models.multiplexing_pretraining_utils import (
    SequenceClassifierOutputMuxed,
    RetrievalHeadIndexPosDemultiplexingBERT,
    RetrievalHeadIndexDemultiplexing,
    IndexPosDemuxModule,
    IndexPosDemultiplexerTokenLevel,
    IndexDemultiplexerTokenLevel,
)
from typing import Optional, Tuple
import math
from transformers.models.electra.modeling_electra import (
    ElectraLayer,
)
import math
from transformers.activations import gelu
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MuxedBertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(config)
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexingBERT(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        d_model = config.hidden_size
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
        elif (
            self.muxing_variant == "gaussian_attention_v2"
        ):
            self.muxing_attention = ElectraLayer(config)
            self.muxing_attention_2 = ElectraLayer(config)
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
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

        self.bert = BertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_output_embeddings(self):
        return self.retrieval_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.retrieval_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
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
        embedding_output = self.bert.embeddings(
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
        elif self.muxing_variant == "gaussian_attention_v2":

            # apply attention to all the sequences to get contextual embeddings. In addition, have a residual connection to the original embeddings
            # once we get the contextual embeddings, apply gaussian hadamard product
            # we can then apply another layer of attention on these embeddings to get weight the contextual representations; we can add a residual connection to the original embeddings
            # finally, we can average across the instances to get the final embeddings
            # apply attention layer across all instances to get contextual embeddings; add residual connection to original embeddings
            embedding_output_attention_1 = self.muxing_attention(
                embedding_output,
                attention_mask=self.get_extended_attention_mask(
                    attention_mask, input_shape, device=input_ids.device
                ),
            )
            embedding_output_attention_1 = embedding_output_attention_1[0]
            embedding_output_attention_1 = (
                embedding_output_attention_1 + embedding_output
            )

            # apply gaussian hadamard product to contextual embeddings

            embedding_output_attention_1_reshaped = embedding_output_attention_1.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_attention_1_reshaped * instance_embed.unsqueeze(0)
            )
            embedding_output_attention_2_input = embedding_output_intermediate.permute(
                0, 2, 1, 3
            )
            embedding_output_attention_2_input = (
                embedding_output_attention_2_input.reshape(
                    -1, num_instances, embedding_dim
                )
            )
            embedding_output_attention_1 = embedding_output_intermediate.view(
                -1, modified_seq_length, embedding_dim
            )

            # apply attention layer across contextual embeddings in each position; add residual connection

            embedding_output_attention_2 = self.muxing_attention_2(
                embedding_output_attention_2_input
            )
            embedding_output_attention_2 = embedding_output_attention_2[0]
            embedding_output_attention_2 = embedding_output_attention_2.view(
                -1, modified_seq_length, num_instances, embedding_dim
            )
            embedding_output_attention_2 = embedding_output_attention_2.permute(
                0, 2, 1, 3
            )
            embedding_output_attention_2 = embedding_output_attention_2.reshape(
                -1, modified_seq_length, embedding_dim
            )

            embedding_output_attention = (
                embedding_output_attention_2 + embedding_output_attention_1
            )
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
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

        outputs = self.bert(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
            output_hidden_states=True,
            output_attentions=True,
        )

        sequence_output = outputs[0]

        if self.retrieval_loss_coeff > 0:
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
            retrieval_predictions = self.retrieval_head(
                sequence_output, instance_labels
            )

        if self.demuxing_variant == "index":
            labels = torch.cat(
                [
                    torch.ones(
                        labels.shape[0],
                        num_instances + 1,
                        device=input_ids.device,
                        dtype=torch.long,
                    ) * -100,
                    labels[:, : -(num_instances + 1)],
                ],
                dim=1,
            )
        is_mlm_applied = labels != -100
        mlm_logits = self.retrieval_head(sequence_output, None, mlm_mask=is_mlm_applied)
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

        if self.retrieval_loss_coeff > 0:
            retrieval_loss_fct = torch.nn.CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(
                retrieval_predictions.view(-1, self.config.vocab_size),
                retrieval_labels.view(-1),
            )

        # make logits align with the length of the input sequence
        # logits = logits[:, special_tokens_end_position:]
        if labels is not None:
            labels = labels[: (modified_batch_size * num_instances)]
            loss_fct = torch.nn.CrossEntropyLoss()
            task_loss = loss_fct(
                logits.reshape(-1, self.config.vocab_size), labels.view(-1)
            )
            if self.retrieval_loss_coeff > 0:
                loss = (self.task_loss_coeff * task_loss) + (
                    self.retrieval_loss_coeff * retrieval_loss
                )
            else:
                loss = self.task_loss_coeff * task_loss

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MuxedBertMaskedLMOutput(
            loss=loss,
            logits=logits,
            task_loss=task_loss,
            retrieval_loss=retrieval_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class MuxedBertMaskedLMOutput(MaskedLMOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    retrieval_loss: torch.FloatTensor = None
    task_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MuxedBertForSequenceClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(
                config
            )
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexingBERT(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        d_model = config.hidden_size
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
        elif self.muxing_variant == "gaussian_attention_v2":
            self.muxing_attention = ElectraLayer(config)
            self.muxing_attention_2 = ElectraLayer(config)
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
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

        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.init_weights()

    def get_output_embeddings(self):
        return self.retrieval_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.retrieval_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
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
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + 680 
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
        embedding_output = self.bert.embeddings(
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
        elif self.muxing_variant == "gaussian_attention_v2":

            # apply attention to all the sequences to get contextual embeddings. In addition, have a residual connection to the original embeddings
            # once we get the contextual embeddings, apply gaussian hadamard product
            # we can then apply another layer of attention on these embeddings to get weight the contextual representations; we can add a residual connection to the original embeddings
            # finally, we can average across the instances to get the final embeddings
            # apply attention layer across all instances to get contextual embeddings; add residual connection to original embeddings
            embedding_output_attention_1 = self.muxing_attention(
                embedding_output,
                attention_mask=self.get_extended_attention_mask(
                    attention_mask, input_shape, device=input_ids.device
                ),
            )
            embedding_output_attention_1 = embedding_output_attention_1[0]
            embedding_output_attention_1 = (
                embedding_output_attention_1 + embedding_output
            )

            # apply gaussian hadamard product to contextual embeddings

            embedding_output_attention_1_reshaped = embedding_output_attention_1.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_attention_1_reshaped * instance_embed.unsqueeze(0)
            )
            embedding_output_attention_2_input = embedding_output_intermediate.permute(
                0, 2, 1, 3
            )
            embedding_output_attention_2_input = (
                embedding_output_attention_2_input.reshape(
                    -1, num_instances, embedding_dim
                )
            )
            embedding_output_attention_1 = embedding_output_intermediate.view(
                -1, modified_seq_length, embedding_dim
            )

            # apply attention layer across contextual embeddings in each position; add residual connection
            attention_mask_attention_2 = attention_mask.view(
                modified_batch_size, self.num_instances, modified_seq_length
            )
            attention_mask_attention_2 = attention_mask_attention_2.permute(0, 2, 1)
            attention_mask_attention_2 = attention_mask_attention_2.reshape(
                -1, self.num_instances
            )
            embedding_output_attention_2 = self.muxing_attention_2(
                embedding_output_attention_2_input,
                self.get_extended_attention_mask(
                    attention_mask_attention_2,
                    attention_mask_attention_2.shape,
                    device=input_ids.device,
                ),
            )
            embedding_output_attention_2 = embedding_output_attention_2[0]
            embedding_output_attention_2 = embedding_output_attention_2.view(
                -1, modified_seq_length, num_instances, embedding_dim
            )
            embedding_output_attention_2 = embedding_output_attention_2.permute(
                0, 2, 1, 3
            )
            embedding_output_attention_2 = embedding_output_attention_2.reshape(
                -1, modified_seq_length, embedding_dim
            )

            embedding_output_attention = (
                embedding_output_attention_2 + embedding_output_attention_1
            )
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
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

        outputs = self.bert(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.demuxing_variant != "index":
            demuxed_sequence_output = self.demultiplexer(sequence_output[:, 0:1, :])
        else:
            demuxed_sequence_output = self.demultiplexer(sequence_output)
            demuxed_sequence_output = demuxed_sequence_output[:, num_instances:num_instances+1, :]
        # demuxed_sequence_output = self.demultiplexer(sequence_output[:, 0:1, :])
        # demuxed_sequence_output = demuxed_sequence_output.squeeze(1)
        demuxed_sequence_output = demuxed_sequence_output.squeeze(1)
        logits = self.classifier(self.dropout(demuxed_sequence_output))
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
                loss_fct = torch.nn.MSELoss()
                task_loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                task_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_fct = torch.nn.CrossEntropyLoss()
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


class MuxedBertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.config = config

        self.num_instances = config.num_instances
        self.muxing_variant = config.muxing_variant
        self.demuxing_variant = config.demuxing_variant
        self.retrieval_loss_coeff = config.retrieval_loss_coeff
        self.task_loss_coeff = config.task_loss_coeff

        if config.demuxing_variant == "index":
            self.demultiplexer = IndexDemultiplexerTokenLevel(config)
            self.retrieval_head = RetrievalHeadIndexDemultiplexing(
                config
            )
        elif config.demuxing_variant == "index_pos":
            self.demux_module = IndexPosDemuxModule(config)
            self.demultiplexer = IndexPosDemultiplexerTokenLevel(
                config, self.demux_module
            )
            self.retrieval_head = RetrievalHeadIndexPosDemultiplexingBERT(
                config, self.demux_module
            )
        else:
            raise NotImplementedError()

        d_model = config.hidden_size
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
        elif self.muxing_variant == "gaussian_attention_v2":
            self.muxing_attention = ElectraLayer(config)
            self.muxing_attention_2 = ElectraLayer(config)
            instance_embedding = random_encoding(
                self.num_instances, d_model, norm=config.gaussian_hadamard_norm
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
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.init_weights()

    def get_output_embeddings(self):
        return self.retrieval_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.retrieval_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

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
                -(torch.arange(num_instances, device=input_ids.device) + 2)
                + 680 
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
        embedding_output = self.bert.embeddings(
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
        elif self.muxing_variant == "gaussian_attention_v2":

            # apply attention to all the sequences to get contextual embeddings. In addition, have a residual connection to the original embeddings
            # once we get the contextual embeddings, apply gaussian hadamard product
            # we can then apply another layer of attention on these embeddings to get weight the contextual representations; we can add a residual connection to the original embeddings
            # finally, we can average across the instances to get the final embeddings
            # apply attention layer across all instances to get contextual embeddings; add residual connection to original embeddings
            embedding_output_attention_1 = self.muxing_attention(
                embedding_output,
                attention_mask=self.get_extended_attention_mask(
                    attention_mask, input_shape, device=input_ids.device
                ),
            )
            embedding_output_attention_1 = embedding_output_attention_1[0]
            embedding_output_attention_1 = (
                embedding_output_attention_1 + embedding_output
            )

            # apply gaussian hadamard product to contextual embeddings

            embedding_output_attention_1_reshaped = embedding_output_attention_1.view(
                modified_batch_size,
                num_instances,
                modified_seq_length,
                embedding_dim,
            )
            instance_embed = self.instance_embedding[:num_instances, :]
            instance_embed = instance_embed.unsqueeze(1).expand(
                num_instances, modified_seq_length, embedding_dim
            )
            embedding_output_intermediate = (
                embedding_output_attention_1_reshaped * instance_embed.unsqueeze(0)
            )
            embedding_output_attention_2_input = embedding_output_intermediate.permute(
                0, 2, 1, 3
            )
            embedding_output_attention_2_input = (
                embedding_output_attention_2_input.reshape(
                    -1, num_instances, embedding_dim
                )
            )
            embedding_output_attention_1 = embedding_output_intermediate.view(
                -1, modified_seq_length, embedding_dim
            )

            # apply attention layer across contextual embeddings in each position; add residual connection

            embedding_output_attention_2 = self.muxing_attention_2(
                embedding_output_attention_2_input
            )
            embedding_output_attention_2 = embedding_output_attention_2[0]
            embedding_output_attention_2 = embedding_output_attention_2.view(
                -1, modified_seq_length, num_instances, embedding_dim
            )
            embedding_output_attention_2 = embedding_output_attention_2.permute(
                0, 2, 1, 3
            )
            embedding_output_attention_2 = embedding_output_attention_2.reshape(
                -1, modified_seq_length, embedding_dim
            )

            embedding_output_attention = (
                embedding_output_attention_2 + embedding_output_attention_1
            )
            embedding_output_attention = embedding_output_attention.view(
                modified_batch_size, num_instances, modified_seq_length, embedding_dim
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

        outputs = self.bert(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids,
            inputs_embeds=embedding_output,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        demuxed_sequence_output = self.demultiplexer(sequence_output)
        demuxed_sequence_output = demuxed_sequence_output.squeeze(1)
        logits = self.classifier(self.dropout(demuxed_sequence_output))

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

            retrieval_predictions = self.retrieval_head(
                sequence_output, instance_labels
            )
        retrieval_loss = None
        task_loss = None
        loss = None
        if dummy_samples_added != 0:
            logits = logits[:-dummy_samples_added]
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            logits = logits[:, special_tokens_end_position:, :]
            task_loss = loss_fct(logits.reshape(-1, self.num_labels), labels.view(-1))
            loss_fct = torch.nn.CrossEntropyLoss()
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
