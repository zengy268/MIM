### ！！！ Still needs to be re-arranged and checked!! Will be done very soon

import os
import torch
import torch.nn.functional as F
from torch import nn
from global_configs import config


os.environ["CUDA_VISIBLE_DEVICES"] = config.device
device = 'cuda'

class Loss_fct(nn.Module):
    def __init__(self):
        super(Loss_fct, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, label):
        loss = self.loss(pred.view(-1), label.view(-1)) 
        return loss


class Loss_intra(nn.Module):
    def __init__(self):
        super(Loss_intra, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, loss_masked, loss_unmasked, output_masked.view(-1), output_unmasked.view(-1)):

        loss = self.loss_penalty(outputl_masked.detach().view(-1), outputl_unmasked.view(-1)) * (loss_l_masked < loss_l_unmasked) \
             + self.loss_penalty(outputl_masked.view(-1), outputl_unmasked.detach().view(-1)) * (loss_l_masked > loss_l_unmasked)
        return loss


class Adjuster(nn.Module):
    def __init__(self):
        super(Adjuster, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, loss_masked, loss_unmasked, output_masked.view(-1), output_unmasked.view(-1)):
        # total_loss = loss_a_masked.item() + loss_v_masked.item() + loss_l_masked.item()
        # lambda_a = loss_a_masked.item() / total_loss
        # lambda_l = loss_l_masked.item() / total_loss
        # lambda_v = loss_v_masked.item() / total_loss

        # denominator = (1 + lambda_a) * (1 + lambda_l) + (1 + lambda_v) * (1 + lambda_l) + (1 + lambda_a) * (
        #             1 + lambda_v)
        # alpha_l = 3 * ((1 + lambda_a) * (1 + lambda_v)) / denominator
        # alpha_a = 3 * ((1 + lambda_l) * (1 + lambda_v)) / denominator
        # alpha_v = 3 * ((1 + lambda_l) * (1 + lambda_a)) / denominator

        # loss_l_inter = loss_kd(masked_l.detach(), masked_a, masked_v, alpha_l) * (loss_l_masked < loss_a_masked) * (loss_l_masked < loss_v_masked)
        # loss_a_inter = loss_kd(masked_a.detach(), masked_l, masked_v, alpha_a) * (loss_a_masked < loss_l_masked) * (loss_a_masked < loss_v_masked)
        # loss_v_inter = loss_kd(masked_v.detach(), masked_l, masked_a, alpha_v) * (loss_v_masked < loss_l_masked) * (loss_v_masked < loss_a_masked)
        # loss = torch.mean(loss_l_inter + loss_a_inter + loss_v_inter)  * self.weight

        return loss
