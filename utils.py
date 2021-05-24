import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Recorder(object):
    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in self.names:
            self.record[name] = []

    def update(self, vals):
        for name, val in zip(self.names, vals):
            self.record[name].append(val)

class ConfusionMeter(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.conf = np.zeros((n_classes, n_classes), dtype=np.int32)

    def update(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        assert y_pred.shape == y_true.shape
        y_pred = y_pred[y_true >= 0]
        y_true = y_true[y_true >= 0]
        x = y_pred.flatten() + self.n_classes * y_true.flatten()
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.n_classes ** 2)
        conf = bincount_2d.reshape((self.n_classes, self.n_classes))
        self.conf += conf

    def value(self):
        return self.conf


class Metric(object):
    def __init__(self, confusion_matrix):
        self.conf = confusion_matrix
        self.true_positive = np.diag(self.conf)
        self.false_positive = np.sum(self.conf, 0) - self.true_positive
        self.false_negative = np.sum(self.conf, 1) - self.true_positive

    def iou(self):
        return self.true_positive / (self.true_positive + self.false_positive + self.false_negative)

    def miou(self):
        return np.nan_to_num(self.iou()).mean()

    def precision(self):
        return self.true_positive / (self.true_positive + self.false_positive)

    def recall(self):
        return self.true_positive / (self.true_positive + self.false_negative)

    def accuracy(self):
        return self.true_positive.sum()/self.conf.sum()

def get_bool(prompt):
    while True:
        try:
           return {"y":True,"n":False}[input(prompt).lower()]
        except KeyError:
           print("Invalid input please enter y/n!")


if __name__ == '__main__':
    a = np.random.randint(0,100,size=(5,5))
    metric = Metric(a)
    print(metric.accuracy())
    print(metric.iou())
    print(metric.precision())
    print(metric.recall())
    print(metric.miou())