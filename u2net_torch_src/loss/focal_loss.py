import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    # def forward(self, input, target):
    #     if input.dim()>2:
    #         input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
    #         input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
    #         input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    #     target = target.view(-1,1).long()

    #     logpt = F.log_softmax(input, 1)
    #     logpt = logpt.gather(1,target)
    #     logpt = logpt.view(-1)
    #     pt = Variable(logpt.data.exp())

    #     if self.alpha is not None:
    #         if self.alpha.type()!=input.data.type():
    #             self.alpha = self.alpha.type_as(input.data)
    #         at = self.alpha.gather(0,target.data.view(-1))
    #         logpt = logpt * Variable(at)

    #     loss = -1 * (1-pt)**self.gamma * logpt
    #     if self.size_average: return loss.mean()
    #     else: return loss.sum()

    def forward(self, input, target, classes='all'):
        if classes in ['all']:
            if input.dim()>2:
                input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1,1).long()

            logpt = F.log_softmax(input, 1)
            logpt = logpt.gather(1,target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                if self.alpha.type()!=input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0,target.data.view(-1))
                logpt = logpt * Variable(at)

            loss = -1 * (1-pt)**self.gamma * logpt
            if self.size_average: return loss.mean()
            else: return loss.sum()

        elif isinstance(classes, list):
            if input.dim()>2:
                input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            # target = target.view(-1,1).long()

            # alpha is pixel-wise weight??
            if self.alpha is not None:
                if self.alpha.type()!=input.data.type():
                    self.alpha = self.alpha.type_as(input.data)

            logpt = F.log_softmax(input, 1)

            # borrow from lovasz_loss
            C = input.size(1)
            loss = []
            class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes

            for c in class_to_sum:
                fg = (target == c).float() # foreground for class c
                class_pred = logpt[:,c]
                if (classes is 'present' and fg.sum() == 0):
                    continue
                logpt = torch.dot(class_pred, fg)
                if self.alpha is not None:
                    class_alpha = (self.alpha == c).float()
                    logpt *= class_alpha
                pt = logpt.data.exp()
                class_loss = -1 * (1-pt)**self.gamma * logpt
                loss.append(class_loss)
            loss = torch.tensor(loss).type(input.dtype)
            if self.size_average: 
                return loss.mean()
            else: 
                return loss.sum()

            