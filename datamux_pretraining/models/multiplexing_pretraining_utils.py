import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_outputs import SequenceClassifierOutput, ModelOutput
from transformers.activations import gelu

class IndexPosDemuxModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_instance_embedding = torch.nn.Parameter(
            torch.randn(config.num_instances, config.hidden_size)
        )

        num_hidden_demux_layers = (
            config.num_hidden_demux_layers
            if hasattr(config, "num_hidden_demux_layers")
            else 2
        )
        config.num_hidden_demux_layers = num_hidden_demux_layers
        self.dense_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        for demux_hidden_idx in range(2, config.num_hidden_demux_layers + 1):
            setattr(
                self,
                f"dense_{demux_hidden_idx}",
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            setattr(
                self,
                f"dropout_{demux_hidden_idx}",
                nn.Dropout(config.hidden_dropout_prob),
            )
            setattr(
                self,
                f"layer_norm_{demux_hidden_idx}",
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            )


class IndexDemultiplexerTokenLevel(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features, **kwargs):

        # extract the first <num sentence> representations and concatenate with the right word
        batch, seqlength, feature_dim = features.shape
        positional_representations = features[:, : self.num_instances, :]
        # concatenate features with the sentence representations based on sentence_labels
        # don't overwrite sentence labels !!

        # need to expand the batch to the original size, need to make predictions
        # on the original
        positional_representations = positional_representations.unsqueeze(2).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = features.unsqueeze(1).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = torch.cat([positional_representations, features], dim=3)
        # increase the batch size by collapsing the first 2 dimensions
        features = features.view(-1, seqlength, 2 * feature_dim)
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        return x


class IndexPosDemultiplexerTokenLevel(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_module = demux_module
        self.pre_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_decoder_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        num_hidden_demux_layers = (
            config.num_hidden_demux_layers
            if hasattr(config, "num_hidden_demux_layers")
            else 2
        )
        self.num_hidden_demux_layers = num_hidden_demux_layers

    def forward(self, features, **kwargs):

        # extract the first <num sentence> representations and concatenate with the right word
        batch, seqlength, feature_dim = features.shape
        positional_representations = (
            self.demux_module.demux_instance_embedding.unsqueeze(0)
        )
        # concatenate features with the sentence representations based on sentence_labels
        # don't overwrite sentence labels !!

        # need to expand the batch to the original size, need to make predictions
        # on the original
        positional_representations = positional_representations.unsqueeze(2).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = features.unsqueeze(1).expand(
            batch, self.num_instances, seqlength, feature_dim
        )
        features = torch.cat([positional_representations, features], dim=3)
        # increase the batch size by collapsing the first 2 dimensions
        features = features.view(-1, seqlength, 2 * feature_dim)

        x = features
        # run the demux module
        # skip the last layer, one specific layer for the retrieval loss
        for demux_hidden_idx in range(1, self.num_hidden_demux_layers):
            dense_cur = getattr(self.demux_module, f"dense_{demux_hidden_idx}")
            layernorm_cur = getattr(self.demux_module, f"dropout_{demux_hidden_idx}")
            x_in = x
            x = dense_cur(x)
            x = gelu(x)
            x = layernorm_cur(x)
            # add skip connection if not first layer
            if demux_hidden_idx > 1:
                x = x + x_in

        # project back to the label space
        x_in = x
        x = self.pre_decoder(x)
        x = gelu(x)
        x = self.pre_decoder_layer_norm(x)
        x = x + x_in
        return x


class RetrievalHeadIndexDemultiplexing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_instances = config.num_instances
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, instance_labels, mlm_mask=None, **kwargs):
        if mlm_mask is None:
            # extract the first <num instance> representations and concatenate with the right word
            batch, seqlength, _ = features.shape
            positional_representations = features[:, : self.num_instances, :]
            # concatenate features with the instance representations based on instance labels
            instance_labels_copy = instance_labels.clone()
            instance_labels_copy[instance_labels == -100] = 0
            positional_embeds = positional_representations[
                torch.arange(batch, device=features.device)
                .unsqueeze(1)
                .repeat(1, seqlength),
                instance_labels_copy,
            ]
            features = torch.cat([positional_embeds, features], dim=2)
            x = self.dense(features)
            x = gelu(x)
            x = self.layer_norm(x)

            # project back to size of vocabulary with bias
            x = self.decoder(x)

            return x
        else:
            # extract the first <num instance> representations and concatenate with the right word
            batch, seqlength, feature_dim = features.shape
            positional_representations = features[:, : self.num_instances, :]
            # concatenate features with the instance representations based on instance labels
            positional_representations = positional_representations.unsqueeze(2).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = features.unsqueeze(1).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = torch.cat([positional_representations, features], dim=3)
            features = features.view(-1, seqlength, 2 * feature_dim)

            # only demux the features that are masked
            features = features[mlm_mask]
            x = self.dense(features)
            x = gelu(x)
            x = self.layer_norm(x)

            # project back to size of vocabulary with bias
            x = self.decoder(x)

            return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class RetrievalHeadIndexPosDemultiplexing(nn.Module):
    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_module = demux_module

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        self.pre_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_decoder_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.num_hidden_demux_layers = config.num_hidden_demux_layers

    def forward(self, features, instance_labels, mlm_mask=None, **kwargs):
        if mlm_mask is None:
            batch, seqlength, _ = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0).expand(
                    batch, -1, -1
                )
            )
            # concatenate features with the instance representations based on instance labels
            instance_labels_copy = instance_labels.clone()
            instance_labels_copy[instance_labels == -100] = 0
            positional_embeds = positional_representations[
                torch.arange(batch, device=features.device)
                .unsqueeze(1)
                .repeat(1, seqlength),
                instance_labels_copy,
            ]
            features = torch.cat([positional_embeds, features], dim=2)
            x = features
            for demux_hidden_idx in range(1, self.num_hidden_demux_layers + 1):
                cur_dense = getattr(self.demux_module, f"dense_{demux_hidden_idx}")
                cur_layer_norm = getattr(
                    self.demux_module, f"layer_norm_{demux_hidden_idx}"
                )
                x_in = x
                x = cur_dense(x)
                x = gelu(x)
                x = cur_layer_norm(x)
                if demux_hidden_idx > 1:
                    x = x + x_in

            x_in = x
            # project back to the label space
            x = self.pre_decoder(x)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)
            x = x + x_in
            x = self.decoder(x)

            return x
        else:
            batch, seqlength, feature_dim = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0)
            )

            positional_representations = positional_representations.unsqueeze(2).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = features.unsqueeze(1).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = torch.cat([positional_representations, features], dim=3)
            features = features.view(-1, seqlength, 2 * feature_dim)

            # only demux the features that are masked
            features = features[mlm_mask]

            x = features
            for demux_hidden_idx in range(1, self.num_hidden_demux_layers + 1):
                cur_dense = getattr(self.demux_module, f"dense_{demux_hidden_idx}")
                cur_layer_norm = getattr(
                    self.demux_module, f"layer_norm_{demux_hidden_idx}"
                )
                x_in = x
                x = cur_dense(x)
                x = gelu(x)
                x = cur_layer_norm(x)
                if demux_hidden_idx > 1:
                    x = x + x_in

            x_in = x
            # project back to the label space
            x = self.pre_decoder(x)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)
            x = x + x_in
            x = self.decoder(x)

            return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class RetrievalHeadIndexPosDemultiplexingBERT(nn.Module):
    def __init__(self, config, demux_module):
        super().__init__()
        self.num_instances = config.num_instances
        self.demux_module = demux_module
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.pre_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_decoder_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.config = config

    def forward(self, features, instance_labels, mlm_mask=None, **kwargs):
        # extract the first <num instance> representations and concatenate with the right word
        if mlm_mask is None:
            batch, seqlength, _ = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0).expand(
                    batch, -1, -1
                )
            )
            # concatenate features with the instance representations based on instance labels
            instance_labels_copy = instance_labels.clone()
            instance_labels_copy[instance_labels == -100] = 0
            positional_embeds = positional_representations[
                torch.arange(batch, device=features.device)
                .unsqueeze(1)
                .repeat(1, seqlength),
                instance_labels_copy,
            ]
            features = torch.cat([positional_embeds, features], dim=2)

            x = self.demux_module.dense_1(features)
            x = gelu(x)
            x = self.demux_module.layer_norm_1(x)
            x = self.demux_module.dense_2(x)
            x = gelu(x)
            x = self.demux_module.layer_norm_2(x)

            # project back to the label space
            x = self.pre_decoder(x)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)
            x = self.decoder(x)

            return x

        else:
            batch, seqlength, feature_dim = features.shape
            positional_representations = (
                self.demux_module.demux_instance_embedding.unsqueeze(0)
            )

            positional_representations = positional_representations.unsqueeze(2).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = features.unsqueeze(1).expand(
                batch, self.num_instances, seqlength, feature_dim
            )
            features = torch.cat([positional_representations, features], dim=3)
            features = features.view(-1, seqlength, 2 * feature_dim)

            # only demux the features that are masked
            features = features[mlm_mask]
            x = self.demux_module.dense_1(features)
            x = gelu(x)
            x = self.demux_module.layer_norm_1(x)
            x = self.demux_module.dense_2(x)
            x = gelu(x)
            x = self.demux_module.layer_norm_2(x)

            # project back to the label space
            x = self.pre_decoder(x)
            x = gelu(x)
            x = self.pre_decoder_layer_norm(x)
            x = self.decoder(x)
            return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class DataCollatorForLanguageModelingMuxed(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, max_pad_tokens=0):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.max_pad_tokens = max_pad_tokens

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # potentially add pad tokens at the end of the sequence
        batch, seq_len = labels.shape
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        pad_lens = torch.randint(0, self.max_pad_tokens + 1, (batch,))
        non_pad_lens = seq_len - pad_lens
        non_pad_attn_mask = gen_attn_mask(non_pad_lens, seq_len)
        pad_attn_mask = ~non_pad_attn_mask

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & non_pad_attn_mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        inputs[pad_attn_mask] = pad_token_id
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorElectra(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, max_pad_tokens=0):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.max_pad_tokens = max_pad_tokens

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # potentially add pad tokens at the end of the sequence
        batch, seq_len = labels.shape
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        pad_lens = torch.randint(0, self.max_pad_tokens + 1, (batch,))
        non_pad_lens = seq_len - pad_lens
        non_pad_attn_mask = gen_attn_mask(non_pad_lens, seq_len)
        pad_attn_mask = ~non_pad_attn_mask

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & non_pad_attn_mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        inputs[pad_attn_mask] = pad_token_id
        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorForLanguageModelingMuxed(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, max_pad_tokens=0):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.max_pad_tokens = max_pad_tokens

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # potentially add pad tokens at the end of the sequence
        batch, seq_len = labels.shape
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        pad_lens = torch.randint(0, self.max_pad_tokens + 1, (batch,))
        non_pad_lens = seq_len - pad_lens
        non_pad_attn_mask = gen_attn_mask(non_pad_lens, seq_len)
        pad_attn_mask = ~non_pad_attn_mask

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & non_pad_attn_mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        inputs[pad_attn_mask] = pad_token_id
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class SequenceClassifierOutputMuxed(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    task_loss: Optional[torch.FloatTensor] = None
    retrieval_loss: Optional[torch.FloatTensor] = None
    retrieval_predictions: Optional[torch.FloatTensor] = None
    retrieval_instance_labels: Optional[torch.FloatTensor] = None
