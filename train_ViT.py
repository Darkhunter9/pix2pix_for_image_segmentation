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

from model_ViT import ViT_cls


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, model, model_dis, criterion, criterion_dis, criterion_l1, optimizer, optimizer_dis, lr_scheduler):
    loss_meter = AverageMeter('train loss')
    acc_meter = AverageMeter('train accuracy')
    miou_meter = AverageMeter('train mIOU')
    conf_meter = ConfusionMeter(cfg.NUM_CLASS)

    lossD_real_meter = AverageMeter('discriminator real loss')
    lossD_fake_meter = AverageMeter('discriminator fake loss')
    lossG_meter = AverageMeter('generator loss')
    dis_penalty_meter = AverageMeter('discriminator penalty')

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

        # update multiple times inside discriminator
        for d_iter in range(cfg.GAN.DIS_ITER):
            if cfg.GAN.WITH_SRC:
                dis_input_real = torch.cat((images, targets_input), 1)
                dis_outputs_real = model_dis(dis_input_real.float())
            else:
                dis_outputs_real = model_dis(targets_input.float())
            # real y, set to one
            y_real = torch.ones(dis_outputs_real.shape).to(DEVICE)
            # print("Dis net output shape", dis_outputs_real.shape)
            loss_dis_real = criterion_dis(dis_outputs_real, y_real)

            # print("loss_dis_real", d_iter+1, loss_dis_real)

            # update discriminator with fake results
            if cfg.GAN.WITH_SRC:
                dis_input_fake = torch.cat((images, outputs), 1)
                dis_outputs_fake = model_dis(dis_input_fake.float())
            else:
                dis_outputs_fake = model_dis(outputs)
            # fake y, set to zero
            y_fake = torch.zeros(dis_outputs_fake.shape).to(DEVICE)
            loss_dis_fake = criterion_dis(dis_outputs_fake, y_fake)

            # print("loss_dis_fake", d_iter+1, loss_dis_fake)

            # sum the two loss
            loss_dis = (loss_dis_real + loss_dis_fake) / 2
            optimizer_dis.zero_grad()
            loss_dis.backward(retain_graph=True)
            optimizer_dis.step()

        lossD_real_meter.update(loss_dis_real.item(), images.size(0))
        lossD_fake_meter.update(loss_dis_fake.item(), images.size(0))

        # update generator weight
        loss = criterion(outputs, targets)
        # print("loss", loss)
        if cfg.GAN.WITH_SRC:
            dis_input_fake = torch.cat((images, outputs), 1)
            dis_outputs_fake = model_dis(dis_input_fake.float())
        else:
            dis_outputs_fake = model_dis(outputs)
        dis_penalty = criterion_dis(dis_outputs_fake, torch.ones(dis_outputs_fake.shape).to(DEVICE))
        dis_penalty_meter.update(dis_penalty.item(), images.size(0))

        # TODO: should include l1 loss?
        loss_l1 = criterion_l1(outputs, targets.unsqueeze(1))
        # print("loss_l1", loss_l1)
        # TODO: summation of the three losses?
        loss_gen = loss + loss_l1 * cfg.GAN.LAMBDA + dis_penalty * cfg.GAN.LAMBDA_DIS
        # print("dis_penalty", dis_penalty)
        lossG_meter.update(loss_gen, images.size(0))

        if i % 50 == 0:
            print("loss_dis_real", loss_dis_real)
            print("loss_dis_fake", loss_dis_fake)
            print("loss_ce", loss)
            print("loss_gen", loss_gen)
            print("dis_penalty", dis_penalty)

        loss_meter.update(loss.item(), images.size(0))
        conf_meter.update(outputs.argmax(1), targets)
        metric = Metric(conf_meter.value())
        acc_meter.update(metric.accuracy())
        miou_meter.update(metric.miou())

        optimizer.zero_grad()
        loss_gen.backward()
        optimizer.step()
        lr_scheduler.step()

    traceGAN = {"lossD_real": lossD_real_meter.avg,
                "lossD_fake": lossD_fake_meter.avg,
                "lossG": lossG_meter.avg,
                "dis_penalty": dis_penalty_meter.avg}
    return loss_meter.avg, acc_meter.avg, miou_meter.avg, traceGAN


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



if __name__ == '__main__':
    experiment_file = "GAN_with_src.yml" # "GAN_with_src.yml"
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
    in_channel = 23 if cfg.GAN.WITH_SRC else 20
    model_dis = ViT_cls(image_size=(512, 512), 
                        patch_size=(16, 16), 
                        num_classes=1,
                        dim=128, 
                        depth=1, 
                        heads=8, 
                        mlp_dim=64, 
                        pool = 'cls', 
                        channels=in_channel, 
                        dim_head=64,
                    )
    model_dis.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    # TODO: discriminator net needs binary cross entropy?
    criterion_dis = torch.nn.BCELoss().to(DEVICE)
    # TODO: should add l1 loss to the generator loss?
    criterion_l1 = nn.L1Loss().to(DEVICE)

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
    recorderGAN = Recorder(["lossD_real", "lossD_fake", "lossG", "dis_penalty"])
    for i in range(cfg.EPOCH):
        print("Epoch", i)
        train_loss, train_acc, train_miou, traceGAN = train(train_loader, model, model_dis, criterion, criterion_dis, criterion_l1, optimizer, optimizer_dis, scheduler)
        print("train_loss:", train_loss)
        print("train_acc:", train_acc)
        print("train_miou:", train_miou)
        val_loss, val_acc, val_miou = validate(val_loader, model, model_dis, criterion)
        print("val_loss:", val_loss)
        print("val_acc:", val_acc)
        print("val_miou:", val_miou)
        recorder.update([train_loss, train_acc, train_miou, val_loss, val_acc, val_miou])
        recorderGAN.update(traceGAN.values())

        if cfg.SAVE:
            torch.save(recorder.record, f"./checkpoint/{save_folder}/trace.log")
            torch.save(recorderGAN.record, f"./checkpoint/{save_folder}/traceGAN.log")
        if cfg.SAVE and val_miou > val_miou_:
            torch.save({
                "epoch": i,
                "model": model,
                "model_dis": model_dis,
                "optimizer": optimizer,
                "scheduler": scheduler,
            }, f"./checkpoint/{save_folder}/state.pth")
            val_miou_ = val_miou
            print("model saved.")
