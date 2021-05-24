import numpy as np
from tqdm import tqdm
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from dataset import get_dataloaders
from utils import AverageMeter, ConfusionMeter, Metric, Recorder, get_bool
from config import get_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
LAMBDA = 1
LAMBDA_DIS = 0.1

def train(train_loader, model, model_dis, criterion, criterion_dis, criterion_l1, optimizer, optimizer_dis, lr_scheduler):
    loss_meter = AverageMeter('train loss')
    acc_meter = AverageMeter('train accuracy')
    miou_meter = AverageMeter('train mIOU')
    conf_meter = ConfusionMeter(cfg.NUM_CLASS)
    model.train()
    model_dis.train()

    for i, (images, targets, _) in enumerate(tqdm(train_loader)):
        # image is input raw image, target is the real results
        targets += 1
        images = images.to(DEVICE)
        targets = targets.long().to(DEVICE)

        # generate fake images
        outputs = model(images)
        # now have both real and fake results, concatenate these and pass into model_dis

        # update discriminiator net with real results
        targets_input = F.one_hot(targets, num_classes=20)
        targets_input = torch.swapaxes(targets_input, 1, 3)
        targets_input = torch.swapaxes(targets_input, 2, 3)
        targets_input = targets_input.to(DEVICE)
        dis_outputs_real = model_dis(targets_input.float())
        # real y, set to one
        y_real = torch.ones(dis_outputs_real.shape).to(DEVICE)
        # print("Dis net output shape", dis_outputs_real.shape)
        loss_dis_real = criterion_dis(dis_outputs_real, y_real)

        print("loss_dis_real", loss_dis_real)

        # update discriminator with fake results
        # model_dis_input_fake = torch.cat((images, outputs), 1)
        dis_outputs_fake = model_dis(outputs.float())
        # fake y, set to zero
        y_fake = torch.zeros(dis_outputs_fake.shape).to(DEVICE)
        loss_dis_fake = criterion_dis(dis_outputs_fake, y_fake)

        print("loss_dis_fake", loss_dis_fake)

        # sum the two loss
        loss_dis = (loss_dis_real + loss_dis_fake) / 2
        optimizer_dis.zero_grad()
        loss_dis.backward(retain_graph=True)
        optimizer_dis.step()

        # # update generator weight
        # bin_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = criterion(outputs, targets)
        print("loss", loss)
        dis_penalty = criterion_dis(torch.zeros(dis_outputs_fake.shape).to(DEVICE), dis_outputs_fake.detach())
        # TODO: should include l1 loss?
        loss_l1 = criterion_l1(outputs, targets.unsqueeze(1))
        print("loss_l1", loss_l1)
        # TODO: summation of the three losses?
        loss_gen = loss + loss_l1 * LAMBDA + dis_penalty * LAMBDA_DIS
        print("dis_penalty", dis_penalty)

        loss_meter.update(loss.item(), images.size(0))
        conf_meter.update(outputs.argmax(1), targets)
        metric = Metric(conf_meter.value())
        acc_meter.update(metric.accuracy())
        miou_meter.update(metric.miou())

        optimizer.zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizer.step()
        lr_scheduler.step()

    return loss_meter.avg, acc_meter.avg, miou_meter.avg


def validate(val_loader, model, model_dis, criterion):
    loss_meter = AverageMeter('validation loss')
    acc_meter = AverageMeter('validation accuracy')
    miou_meter = AverageMeter('validation mIOU')
    conf_meter = ConfusionMeter(cfg.NUM_CLASS)

    with torch.no_grad():
        model.eval()
        for i, (images, targets, _) in enumerate(tqdm(val_loader)):
            images = images.to(DEVICE)
            targets += 1
            targets = targets.long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))
            conf_meter.update(outputs.argmax(1), targets)

            metric = Metric(conf_meter.value())
            acc_meter.update(metric.accuracy())
            miou_meter.update(metric.miou())

    return loss_meter.avg, acc_meter.avg, miou_meter.avg


### Notes:
# this discriminator has input shape 20, 512, 512 because the input is only outputs
class Discriminator_Net(nn.Module):
    def __init__(self):
        super(Discriminator_Net, self).__init__()
        self.net = nn.Sequential(
            # C64
            nn.Conv2d(20, 64, kernel_size=(4, 4), stride=(2, 2), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # C128
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # C256
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 512
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # The last layer, no batchnorm, 
            nn.Conv2d(512, 19, kernel_size=(4, 4), bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.net(input)

if __name__ == '__main__':
    experiment_file = "baseline_0.yml"
    cfg = get_config(f"./experiments/{experiment_file}")
    save_folder = experiment_file.split('.')[0] + "_" + cfg.MODEL
    if cfg.SAVE:
        if os.path.exists(f"./checkpoint/{save_folder}"):
            res = get_bool(f"./checkpoint/{save_folder} already exists. Overwrite? (y/n)")
            if not res: sys.exit(0)
        else:
            os.mkdir(f"./checkpoint/{save_folder}")
            cfg.dump(stream = open(f"./checkpoint/{save_folder}/config.yml", 'w'))

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    if cfg.MODEL == "unet":
        model = smp.Unet(
            encoder_name=cfg.MODEL_ENCODER,
            encoder_weights="imagenet",
            in_channels=3,
            classes=cfg.NUM_CLASS,
        ).to(DEVICE)
    else: raise ValueError(cfg.MODEL)

    # model_dis is the discriminator network
    model_dis = Discriminator_Net()
    model_dis.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    # TODO: discriminator net needs binary cross entropy?
    criterion_dis = torch.nn.BCELoss().to(DEVICE)
    # TODO: should add l1 loss to the generator loss?
    criterion_l1 = nn.L1Loss()

    if cfg.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
        optimizer_dis = torch.optim.AdamW(model_dis.parameters(), lr=cfg.LR)
    else: raise ValueError(cfg.OPTIMIZER)

    if cfg.LR_SCHEDULER == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_RESTART, eta_min=cfg.LR_MIN)
    else: raise ValueError(cfg.LR_SCHEDULER)

    train_loader, val_loader = get_dataloaders(cfg)

    val_miou_ = 0
    recorder = Recorder(["train_loss", "train_acc", "train_miou", "val_loss", "val_acc", "val_miou"])
    for i in range(cfg.EPOCH):
        print("Epoch", i)
        train_loss, train_acc, train_miou = train(train_loader, model, model_dis, criterion, criterion_dis, criterion_l1, optimizer, optimizer_dis, scheduler)
        print("train_loss:", train_loss)
        print("train_acc:", train_acc)
        print("train_miou:", train_miou)
        val_loss, val_acc, val_miou = validate(val_loader, model, model_dis, criterion)
        print("val_loss:", val_loss)
        print("val_acc:", val_acc)
        print("val_miou:", val_miou)
        recorder.update([train_loss, train_acc, train_miou, val_loss, val_acc, val_miou])

        torch.save(recorder.record, f"./checkpoint/{save_folder}/trace.log")
        if cfg.SAVE and val_miou > val_miou_:
            torch.save({
                "epoch": i,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
            }, f"./checkpoint/{save_folder}/state.pth")
            val_miou_ = val_miou
            print("model saved.")

