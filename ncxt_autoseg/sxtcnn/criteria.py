import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight if weight else 1
        self.eps = 1e-6

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)

        encoded_target = softmax.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target_clone = target.clone()
            target_clone[mask] = 0
            encoded_target.scatter_(1, target_clone.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = softmax * encoded_target
        numerator = intersection.sum(0).sum(1).sum(1).sum(1)
        denominator = softmax + encoded_target
        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1).sum(1) + self.eps

        loss_per_channel = self.weight * (numerator / denominator)

        return 1 - 2 / softmax.size(1) * loss_per_channel.sum()


class CrossEntropyLoss_DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, eps=1e-6):
        super(CrossEntropyLoss_DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight if weight else 1
        self.eps = eps
        self.celoss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)

        encoded_target = softmax.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target_clone = target.clone()
            target_clone[mask] = 0
            encoded_target.scatter_(1, target_clone.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = softmax * encoded_target
        numerator = intersection.sum(0).sum(1).sum(1).sum(1)
        denominator = softmax + encoded_target
        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1).sum(1) + self.eps

        loss_per_channel = self.weight * (numerator / denominator)

        return (
            self.celoss(input, target)
            + 1
            - 2 / softmax.size(1) * loss_per_channel.sum()
        )


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.celoss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, input, target):
        return self.celoss(input, target)


class SingleMSELoss(nn.Module):
    def __init__(
        self,
    ):
        super(SingleMSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        target_view = target[:, None, :, :, :]
        return self.loss(input, target_view)
