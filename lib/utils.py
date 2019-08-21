from termcolor import cprint



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur, n=1):
        self.cur = cur
        self.sum += cur * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_lr(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 20 every 10 epochs"""
    lr = lr * (0.2 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def log_print(text, log_file, color = None, on_color = None, attrs = None):
    print(text, file=log_file)
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)






