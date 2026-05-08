import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.SHKPNet import SHKP_Net
from datasets.keypointnet_data import KPSH_Dataset
from loss import HungarianKPLoss


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.train_dataset = KPSH_Dataset(args, True)
        self.val_dataset = KPSH_Dataset(args, False)
        self.test_dataset = KPSH_Dataset(args, False)
        args.num_classes = 2#self.train_dataset.nclasses
        self.net = SHKP_Net(args)
        self.criterion = HungarianKPLoss()

    def forward(self, img_emb, img_feat):
        return self.net.forward(img_emb, img_feat)

    def training_step(self, batch, batch_idx, optimizer_idx):
        kp_c, img, img_emb, img_feat, mask, heats, vis_sts= batch

        probs, pred_heat, pred_vis = self(img_emb, img_feat)
        pred_heat = pred_heat * mask

        loss, loss_c, loss_h, loss_v = self.criterion(heats, vis_sts, probs, pred_heat, pred_vis)

        if optimizer_idx == 0:
            self.log('loss_c', loss_c, batch_size=kp_c.size(0))
            return loss_c

        if optimizer_idx == 1:
            self.log('loss_h', loss_h, batch_size=kp_c.size(0))
            self.log('loss_v', loss_v, batch_size=kp_c.size(0))
            return loss_h + loss_v

    def validation_step(self, batch, _):
        kp_c, img, img_emb, img_feat, mask, heats, vis_sts = batch

        probs, pred_heat, pred_vis = self(img_emb, img_feat)
        pred_heat = pred_heat * mask

        loss, loss_c, loss_h, loss_v = self.criterion(heats, vis_sts, probs, pred_heat, pred_vis)
        self.log('val_loss', loss, True, batch_size=kp_c.size(0))
        self.log('val_loss_c', loss_c, True, batch_size=kp_c.size(0))
        self.log('val_loss_h', loss_h, True, batch_size=kp_c.size(0))
        self.log('val_loss_v', loss_v, True, batch_size=kp_c.size(0))

    def test_step(self, batch, _):
        kp_c, img, img_emb, img_feat, mask, heats, vis_sts = batch

        probs, pred_heat, pred_vis = self(img_emb, img_feat)
        pred_heat = pred_heat * mask

        loss, loss_c, loss_h, loss_v = self.criterion(heats, vis_sts, probs, pred_heat, pred_vis)
        self.log('test_loss', loss, True, batch_size=kp_c.size(0))

    def configure_optimizers(self):
        args = self.hparams.args
        optimizer_heat = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        optimizer_cls = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler_heat = torch.optim.lr_scheduler.OneCycleLR(optimizer_heat, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        scheduler_cls = torch.optim.lr_scheduler.OneCycleLR(optimizer_cls, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer_cls, optimizer_heat], [{'scheduler': scheduler_heat, 'interval': 'step'}, {'scheduler': scheduler_cls, 'interval': 'step'}]

    def train_dataloader(self):
        args = self.hparams.args
        return DataLoader(self.train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(self.val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(self.test_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)


class LitModelInference(LitModel):
    def forward(self, x):
        return torch.argmax(self.net(x), dim=2)