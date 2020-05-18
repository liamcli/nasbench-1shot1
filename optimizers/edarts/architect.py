import numpy as np
import torch
from torch.autograd import Variable

class AdaptiveLR(object):
    def __init__(self, base_lrs, min_lr, max_lr):
        self.base_lrs = base_lrs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.arch_grad_norms = np.zeros(len(base_lrs))

    def update_norm_get_lr(self, i, p):
        value = torch.norm(p, p=float('inf')).item()
        self.arch_grad_norms[i] += value**2
        lr = self.base_lrs[i] / np.sqrt(max(1, self.arch_grad_norms[i]))
        return max(self.min_lr, min(lr, self.max_lr))

class History:
    """
    Data class for saving architecture search history.  
    """

    def __init__(
        self,
        model,
        architect,
        to_save=("alphas", "edges")
    ):

        self.model = model
        self.architect = architect
        self.to_save = to_save
        self.dict = {}

        for field in to_save:
            self.dict[field] = []

    def update_history(self):
        for field in self.to_save:
            if field == "alphas":
                values = self.architect.alphas.data.cpu().numpy()
                self.dict["alphas"].append(values)
            elif field == "edges":
                values = [v.data.cpu().numpy() for v in self.architect.edges]
                values.append(self.architect.output_weights.data.cpu().numpy())
                self.dict["edges"] .append(values)

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def normalize(x, dim):
    x = torch.clamp(x, min=1e-5)
    return x / x.sum(dim=dim, keepdim=True)

class Architect(object):

    def __init__(self, model, args):
        self.alpha_lr = args.arch_learning_rate 
        self.edge_lr = args.edge_learning_rate 
        self.model = model
        self.alphas = model._arch_parameters[0]
        self.output_weights = model._arch_parameters[1]
        self.edges = model._arch_parameters[2:]
        self.history = History(model, self)
        base_lrs = [self.alpha_lr] + [self.edge_lr] * (len(self.edges) + 1)
        self.adapt_lr = False
        self.adaptive_lr = AdaptiveLR(base_lrs, 0.0001, 0.1)
        self.steps = 0
        self.arch_weight_decay = args.arch_weight_decay

    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)


    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.steps += 1
        self.model.zero_grad()
        self._backward_step(input_valid, target_valid)

        for p in self.model._arch_parameters[0:1]:
            if self.adapt_lr:
                norm_inf = max(torch.norm(p.grad.data, p=float('inf'), dim=-1))
                lr = self.alpha_lr / norm_inf
                #lr = self.adaptive_lr.update_norm_get_lr(0, p.grad.data)
            else:
                lr = self.alpha_lr
            if self.steps % 100==0:
                print('operation lr: {}'.format(lr))
            p.data.mul_(torch.exp(-lr * p.grad.data))
            p.data = normalize(p.data, -1)
            p.grad.detach_()
            p.grad.zero_()

        i = 1
        for p in self.model._arch_parameters[1:]:
            if self.adapt_lr:
                #lr = self.adaptive_lr.update_norm_get_lr(i, p.grad.data)
                norm_inf = torch.norm(p.grad.data, p=float('inf'), dim=-1)
                lr = self.edge_lr / norm_inf.item()
            else:
                lr = self.edge_lr
            if self.steps % 100==0:
                print('edge lr {}: {}'.format(i, lr))
            i += 1
            p.data.mul_(torch.exp(-lr * p.grad.data))
            p.data = normalize(p.data, -1)
            p.grad.detach_()
            p.grad.zero_()

    def _backward_step(self, input_valid, target_valid):
        entropic_reg = 0
        for p in self.model._arch_parameters:
            entropic_reg += torch.sum(p * torch.log(p/(1/p.size()[1])))
        loss = self._val_loss(self.model, input_valid, target_valid) + self.arch_weight_decay * entropic_reg
        loss = self._val_loss(self.model, input_valid, target_valid)
        loss.backward()

