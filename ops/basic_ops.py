# import torch
# import math
# from torch.autograd import Variable
#
#
# class Identity(torch.nn.Module):
#     def forward(self, input):
#         return input
#
#
# class SegmentConsensus(torch.autograd.Function):
#
#     # def __init__(self, consensus_type, dim=1):
#     #     self.consensus_type = consensus_type
#     #     self.dim = dim
#     #     self.shape = None
#
#     @staticmethod
#     def forward(ctx, input_tensor):
#         ctx.shape = input_tensor.size()
#         ctx.consensus_type = 'avg'
#         ctx.dim = 1
#         # ctx.dim = ConsensusModule.getdim(ctx)
#
#         if ctx.consensus_type == 'avg':
#             output = input_tensor.mean(dim=ctx.dim, keepdim=True)
#         elif ctx.consensus_type == 'identity':
#             output = input_tensor
#         else:
#             output = None
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         if ctx.consensus_type == 'avg':
#             grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
#         elif ctx.consensus_type == 'identity':
#             grad_in = grad_output
#         else:
#             grad_in = None
#
#         return grad_in
#
#
# class ConsensusModule(torch.nn.Module):
#     # ['avg', 'max', 'topk', 'identity', 'rnn', 'cnn']
#     def __init__(self, consensus_type, dim=1):
#         super(ConsensusModule, self).__init__()
#         self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
#         self.dim = dim
#
#     def getdim(self):
#         return self.dim
#
#     def forward(self, input):
#         return SegmentConsensus.apply(input)


import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)
