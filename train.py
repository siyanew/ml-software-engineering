import argparse
import collections
import random

import numpy as np
import torch

import pytorch.model.loss as module_loss
import pytorch.model.metric as module_metric
from pytorch.parse_config import ConfigParser
from pytorch.trainer import Trainer

# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

# Fix GPU problems by first calling current_device()
# https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()


def main(config: ConfigParser):
    logger = config.get_logger('train')

    # Setup data_loader instances
    data_loader = config.init_obj_from_file('data_loader')

    # Check if the input/output dimensions are the same
    assert len(data_loader.SRC.vocab) == config['arch']['args']['input_dim'], \
        "Input dimensions need to match, check amount of unique tokens in the src vocab"
    assert len(data_loader.TRG.vocab) == config['arch']['args']['output_dim'], \
        "Output dimensions need to match, check amount of unique tokens in the trg vocab"

    # Load torch text Iterator
    if config['data_loader']['iterator']:
        train_data_loader = data_loader.split_train()
        valid_data_loader = data_loader.split_validation()
    else:
        raise NotImplementedError
        # train_data_loader = data_loader
        # valid_data_loader = data_loader.split_validation()

    # Build model architecture, then print to console
    model = config.init_obj_from_file('arch')
    logger.info(model)

    # Get loss criterion function
    criterion = getattr(module_loss, config['loss']['function'])

    # Set the padding index in the criterion such that pad tokens are ignored
    if config['loss']['padding_idx']:
        criterion = criterion(data_loader.TRG.vocab.stoi['<pad>'])

    # Set special token indices in the model for packing
    if 'packed' in config['arch'] and config['arch']['packed']:
        model.set_tokens(data_loader.SRC.vocab.stoi['<pad>'],
                         data_loader.TRG.vocab.stoi['<sos>'],
                         data_loader.TRG.vocab.stoi['<eos>'])

    # Load extra metrics that can be analysed during training
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # Build optimizer and learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Start training
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train a PyTorch model')
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
