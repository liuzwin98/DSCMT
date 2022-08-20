"""
This module provides our implementation of different functions to do video-level classification and stream fusion
"""
import numpy as np
from utils.metrics import softmax
import torch


def default_aggregation_func(score_arr, normalization=True, crop_agg=None):
    """
    This is the default function for make video-level prediction
    :param score_arr: a 3-dim array with (frame, crop, class) layout
    :return:
    """
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(crop_agg(score_arr, axis=1).mean(axis=0))
    else:
        return crop_agg(score_arr, axis=1).mean(axis=0)


def top_k_aggregation_func(score_arr, k, normalization=True, crop_agg=None):
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(np.sort(crop_agg(score_arr, axis=1), axis=0)[-k:, :].mean(axis=0))
    else:
        return np.sort(crop_agg(score_arr, axis=1), axis=0)[-k:, :].mean(axis=0)


def sliding_window_aggregation_func(score, spans=[1, 2, 4, 8, 16], overlap=0.2, norm=True, fps=1):
    """
    This is the aggregation function used for ActivityNet Challenge 2016
    :param score:
    :param spans:
    :param overlap:
    :param norm:
    :param fps:
    :return:
    """
    frm_max = score.max(axis=1)
    slide_score = []

    def top_k_pool(scores, k):
        return np.sort(scores, axis=0)[-k:, :].mean(axis=0)

    for t_span in spans:
        span = t_span * fps
        step = int(np.ceil(span * (1-overlap)))
        local_agg = [frm_max[i: i+span].max(axis=0) for i in xrange(0, frm_max.shape[0], step)]
        k = max(15, len(local_agg)/4)
        slide_score.append(top_k_pool(np.array(local_agg), k))

    out_score = np.mean(slide_score, axis=0)

    if norm:
        return softmax(out_score)
    else:
        return out_score


def default_fusion_func(major_score, other_scores, fusion_weights, norm=True):
    assert len(other_scores) == len(fusion_weights)
    out_score = major_score
    for s, w in zip(other_scores, fusion_weights):
        out_score += s * w

    if norm:
        return softmax(out_score)
    else:
        return out_score


class smooth_crossentropy(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(smooth_crossentropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, gold):
        n_class = pred.size(1)
        one_hot = torch.full_like(pred, fill_value=self.smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - self.smoothing)
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)

        return torch.nn.functional.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction)