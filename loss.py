import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable

class HungarianKPLoss(nn.Module):
    def __init__(self, lambda_cls=100., lambda_hm=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_hm = lambda_hm

    def forward(self, gt_hm, gt_vis, pred_cls, pred_hm, pred_vis):
        B, M, H, W = pred_hm.shape
        _, K_total, _, _ = gt_hm.shape

        total_cls_loss = 0
        total_hm_loss = 0
        total_vis_loss = 0
        valid_batches = 0

        for b in range(B):

            gt_hm_b = gt_hm[b]
            gt_vis_b = gt_vis[b]
            valid_mask = ~(gt_hm_b == -1).all(dim=(1, 2))
            gt_hm_b = gt_hm_b[valid_mask]  # (K_valid, H, W)
            gt_vis_b = gt_vis_b[valid_mask]

            pred_hm_b = pred_hm[b]    # M × H × W
            pred_cls_b = pred_cls[b]  # M × 2
            pred_vis_b = pred_vis[b]

            K_valid = gt_hm_b.shape[0]

            if K_valid > 0:

                hm_cost = ((gt_hm_b[:, None] - pred_hm_b[None]) ** 2).mean(dim=(2, 3))
                log_probs = pred_cls_b.log_softmax(dim=-1)  # (M,2)
                cls_cost = -log_probs[:, 1].unsqueeze(0).expand(K_valid, M)
                cost = self.lambda_cls * cls_cost + self.lambda_hm * hm_cost

                row_ind, col_ind = linear_sum_assignment(cost.detach().cpu())

                cls_target = torch.zeros(M, dtype=torch.long, device=pred_cls.device)
                cls_target[col_ind] = 1

                batch_cls_loss = F.cross_entropy(pred_cls_b, cls_target)
                matched_pred_hm = pred_hm_b[col_ind]        # K × H × W
                matched_gt_hm = gt_hm_b[row_ind]            # K × H × W

                batch_hm_loss = F.smooth_l1_loss(
                    matched_pred_hm, matched_gt_hm
                )

                # vis loss
                matched_pred_vis = pred_vis_b[col_ind]
                matched_gt_vis = gt_vis_b[row_ind]
                batch_vis_loss = F.mse_loss(matched_pred_vis.squeeze(), matched_gt_vis)
            else:
                cls_target = torch.zeros(M, dtype=torch.long, device=pred_cls.device)
                batch_cls_loss = F.cross_entropy(pred_cls_b, cls_target)
                batch_hm_loss = torch.tensor(0.0, device=pred_cls.device)
                batch_vis_loss = torch.tensor(0.0, device=pred_cls.device)

            total_cls_loss += batch_cls_loss
            total_hm_loss += batch_hm_loss
            total_vis_loss += batch_vis_loss
            valid_batches += 1

        loss_cls = total_cls_loss / valid_batches
        loss_hm = total_hm_loss / valid_batches
        loss_vis = total_vis_loss / valid_batches

        loss_total = loss_cls + loss_hm * 1000 + loss_vis

        return loss_total, loss_cls, loss_hm*1000, loss_vis

class FocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.num_classes)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].to(predict.device)
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
