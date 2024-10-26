import time
import torch
import numpy as np
import random
import os
import logging



def get_new_log_dir(root='./logs', prefix='', tag=''):
    """
    :param root: the dir path of log.
    :param prefix: the prefix name of log file.
    :param tag: the tag name of log file
    :return: the path of log dir.
    """
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def seed_all(seed):
    """ Seed. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name, log_dir=None, log_name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        if log_name is not None:
            file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
        else:
            file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)








