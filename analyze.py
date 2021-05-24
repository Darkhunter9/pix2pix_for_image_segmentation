import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
import cv2

from dataset import get_dataloaders
from config import get_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_train_val(train_loss, train_acc, train_miou, val_loss, val_acc, val_miou):
    print(train_loss)
    print(train_miou)
    print(train_acc)
    x = np.arange(len(train_loss))+1
    fig = plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k', frameon=True)

    plt.subplot(131)
    t = plt.plot(x, train_loss, 'b')
    v = plt.plot(x, val_loss, 'r')
    plt.legend([t[0], v[0]], ["train", "val"]) # loc='lower right', fontsize=16, frameon=False
    plt.xlabel("Epoch") # fontsize=14
    plt.title("Loss")

    plt.subplot(132)
    t = plt.plot(x, train_acc, 'b')
    v = plt.plot(x, val_acc, 'r')
    plt.legend([t[0], v[0]], ["train", "val"]) # loc='lower right', fontsize=16, frameon=False
    plt.xlabel("Epoch") # fontsize=14
    plt.title("Accuracy")


    plt.subplot(133)
    t = plt.plot(x, train_miou, 'b')
    v = plt.plot(x, val_miou, 'r')
    plt.legend([t[0], v[0]], ["train", "val"]) # loc='lower right', fontsize=16, frameon=False
    plt.xlabel("Epoch") # fontsize=14
    plt.title("Mean IOU")

    fig.savefig("analysis_epoch_%s.png" %(str(len(train_loss))),dpi=80,bbox_inches='tight')


def get_color_pallete(npimg, dataset='pascal_voc'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')

    Returns
    -------
    out_img : PIL.Image
        Image with color pallete

    """
    cityspallete = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(cityspallete)
    return out_img


def plot_generated_figures(model, loader):
    for i, (images, targets, filenames) in enumerate(train_loader):
        if i > 0:
            break
        fig = plt.figure(num=None, figsize=(3, 6), dpi=160, facecolor='w', edgecolor='k', frameon=True)
        plt.title('Input')
        city = filenames[0].split('_')[0]
        try:
            split = 'train'
            plt.imshow(
                cv2.imread('./data/cityscapes/leftImg8bit/' + split + '/' + city + '/' + filenames[0])[:, :, ::-1])
        except:
            split = 'val'
            plt.imshow(
                cv2.imread('./data/cityscapes/leftImg8bit/' + split + '/' + city + '/' + filenames[0])[:, :, ::-1])

        outputs = model(images.to(DEVICE)).detach()[0].argmax(dim=0)

        outputs = get_color_pallete(outputs.cpu().numpy(), dataset='citys')
        fig = plt.figure(num=None, figsize=(3, 6), dpi=160, facecolor='w', edgecolor='k', frameon=True)
        plt.title('Prediction')
        plt.imshow(outputs)
        plt.show()

        targets = get_color_pallete(targets[0].numpy(), dataset='citys')
        fig = plt.figure(num=None, figsize=(3, 6), dpi=160, facecolor='w', edgecolor='k', frameon=True)
        plt.title('Label')
        plt.imshow(targets)
        plt.show()


if __name__ == "__main__":
    experiment_file = "baseline_0.yml"
    cfg = get_config(f"./experiments/{experiment_file}")
    cfg.defrost()
    cfg.BATCH_SIZE = 1
    cfg.CROP_SIZE = (1024, 2048)
    save_folder = experiment_file.split('.')[0] + "_" + cfg.MODEL

    # parse the loss, acc, miou and make plots
    train_loss, train_acc, train_miou, val_loss, val_acc, val_miou = torch.load(f"./checkpoint/{save_folder}/trace.log").values()
    plot_train_val(train_loss, train_acc, train_miou, val_loss, val_acc, val_miou)

    # load in the model, visualize the figure generator
    # model = torch.load(modelpath)
    model = torch.load(f"./checkpoint/{save_folder}/state.pth")["model"].to(DEVICE)
    model.eval()
    train_loader, val_loader = get_dataloaders(cfg)
    plot_generated_figures(model, train_loader)
    plot_generated_figures(model, val_loader)