import torch
import math
import re
import os
import transformers
import signal
import logging
logger = logging.getLogger(__name__)

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def random_encoding(max_positions, d_model, norm=1):
    gauss = torch.randn((max_positions, d_model))
    gauss = gauss / torch.norm(gauss, dim=1).unsqueeze(1)
    gauss *= norm
    return gauss

def gen_attn_mask(sequence_length, len=None):
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(len)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, len)
    seq_range_expand = seq_range_expand.to(sequence_length.device)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def binary_encoding(max_position, d_model, epsilon=0.3):
    assert epsilon <= 1 and epsilon >= 0, "epsilon value should lie in [0,1)"
    chunk_size = d_model // max_position
    start_of_chunks = chunk_size * torch.arange(max_position)
    end_of_chunks = start_of_chunks + chunk_size
    end_of_chunks[-1] = d_model
    # tweak start and end states to account for epsilon
    num_intersection = (epsilon / 2) * chunk_size
    start_of_chunks[1:] = start_of_chunks[1:] - num_intersection
    end_of_chunks[:-1] = end_of_chunks[:-1] + num_intersection

    # for loop here :( , not worth vectorizing, only called once
    binary_embeds = torch.zeros(max_position, d_model)
    for pos in range(max_position):
        binary_embeds[pos, start_of_chunks[pos] : end_of_chunks[pos]] = 1
    return binary_embeds

def entropy(p):
    """Compute the entropy of a probability distribution"""
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)