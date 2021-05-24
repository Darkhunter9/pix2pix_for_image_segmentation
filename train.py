import numpy as np
from tqdm import tqdm
import os, sys

import torch
import segmentation_models_pytorch as smp

from dataset import get_dataloaders
from utils import AverageMeter, ConfusionMeter, Metric, Recorder, get_bool
from config import get_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, model, criterion, optimizer, lr_scheduler):
    loss_meter = AverageMeter('train loss')
    acc_meter = AverageMeter('train accuracy')
    miou_meter = AverageMeter('train mIOU')
    conf_meter = ConfusionMeter(cfg.NUM_CLASS)
    model.train()

    for i, (images, targets, _) in enumerate(tqdm(train_loader)):
        images = images.to(DEVICE)
        targets = targets.long().to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss_meter.update(loss.item(), images.size(0))
        conf_meter.update(outputs.argmax(1), targets)
        metric = Metric(conf_meter.value())
        acc_meter.update(metric.accuracy())
        miou_meter.update(metric.miou())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    return loss_meter.avg, acc_meter.avg, miou_meter.avg


def validate(val_loader, model, criterion):
    loss_meter = AverageMeter('validation loss')
    acc_meter = AverageMeter('validation accuracy')
    miou_meter = AverageMeter('validation mIOU')
    conf_meter = ConfusionMeter(cfg.NUM_CLASS)

    with torch.no_grad():
        model.eval()
        for i, (images, targets, _) in enumerate(tqdm(val_loader)):
            images = images.to(DEVICE)
            targets = targets.long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))
            conf_meter.update(outputs.argmax(1), targets)

            metric = Metric(conf_meter.value())
            acc_meter.update(metric.accuracy())
            miou_meter.update(metric.miou())

    return loss_meter.avg, acc_meter.avg, miou_meter.avg


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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(DEVICE)

    if cfg.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    else: raise ValueError(cfg.OPTIMIZER)

    if cfg.LR_SCHEDULER == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_RESTART, eta_min=cfg.LR_MIN)
    else: raise ValueError(cfg.LR_SCHEDULER)

    train_loader, val_loader = get_dataloaders(cfg)

    val_miou_ = 0
    recorder = Recorder(["train_loss", "train_acc", "train_miou", "val_loss", "val_acc", "val_miou"])
    for i in range(cfg.EPOCH):
        print("Epoch", i)
        train_loss, train_acc, train_miou = train(train_loader, model, criterion, optimizer, scheduler)
        print("train_loss:", train_loss)
        print("train_acc:", train_acc)
        print("train_miou:", train_miou)
        val_loss, val_acc, val_miou = validate(val_loader, model, criterion)
        print("val_loss:", val_loss)
        print("val_acc:", val_acc)
        print("val_miou:", val_miou)
        recorder.update([train_loss, train_acc, train_miou, val_loss, val_acc, val_miou])

        if cfg.SAVE:
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

