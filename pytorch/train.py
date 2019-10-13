import argparse
import collections
import importlib

import numpy as np
import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Fix GPU problems by first calling current_device()
# https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    if config['data_loader']['iterator']:
        train_data_loader = data_loader.split_train()
        valid_data_loader = data_loader.split_validation()
    else:
        train_data_loader = data_loader
        valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    module_arch = importlib.import_module(config['arch']['file'])
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']['function'])

    # set the padding index in the criterion such that we ignore pad tokens
    if config['loss']['padding_idx']:
        criterion = criterion(data_loader.TRG.vocab.stoi['<pad>'])

    if 'packed' in config['arch'] and config['arch']['packed']:
        model.set_tokens(data_loader.SRC.vocab.stoi['<pad>'],
                         data_loader.TRG.vocab.stoi['<sos>'],
                         data_loader.TRG.vocab.stoi['<eos>'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
