import math
from typing import Callable, List

import numpy as np
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from base import BaseModel, BaseTrainer
from parse_config import ConfigParser
from utils import MetricTracker, inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model: BaseModel, criterion: Module, metric_ftns: List[Callable],
                 optimizer: Optimizer, config: ConfigParser, data_loader,
                 valid_data_loader=None, lr_scheduler: _LRScheduler = None, len_epoch: int = None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', 'perplexity', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'perplexity', writer=self.writer)

        self.step = self.model.process_batch

    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.data_loader):

            self.optimizer.zero_grad()
            output, target = self.step(batch)

            loss = self.criterion(output, target)
            loss.backward()

            # clip the gradients to prevent them from exploding (a common issue in RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        # Track perplexity
        self.train_metrics.update('perplexity', math.exp(self.train_metrics.get_avg('loss')))

        log = dict()
        train_log = self.train_metrics.result()
        log.update(**{'train_' + k: v for k, v in train_log.items()})

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if self.config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
                self.lr_scheduler.step(val_log['loss'])
            else:
                self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch: int) -> dict:
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                output, target = self.step(batch, train=False)

                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # Track perplexity
        self.valid_metrics.update('perplexity', math.exp(self.valid_metrics.get_avg('loss')))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx: int) -> str:
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
