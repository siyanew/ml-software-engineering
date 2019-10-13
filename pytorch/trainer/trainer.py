import math

import numpy as np
import torch

from base import BaseTrainer
from utils import MetricTracker, inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
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

        if 'packed' in config['arch'] and config['arch']['packed']:
            self.step = self._packed_step
        else:
            self.step = self._step

    def _step(self, batch, train=True):
        # Note this only works for the BucketIterator, and not for torch Dataloaders
        src = batch.src
        target = batch.trg

        src, target = src.to(self.device), target.to(self.device)
        if train:
            output = self.model(src, target)
        else:
            # Turn of teacher forcing
            output = self.model(src, target, teacher_forcing_ratio=0)

        # as the loss function only works on 2d inputs with 1d targets we need to flatten each of them with .view
        # we also don't want to measure the loss of the <sos> token, hence we slice off the first column of the
        # output and target tensors

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        output = output[1:].view(-1, output.shape[-1])
        target = target[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        return output, target

    def _packed_step(self, batch, train=True):
        src, src_len = batch.src
        trg = batch.trg

        src, src_len = src.to(self.device), src_len.to(self.device)
        trg = trg.to(self.device)

        if train:
            output, attention = self.model(src, src_len, trg)
        else:
            output, attention = self.model(src, src_len, trg, teacher_forcing_ratio=0)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        return output, trg

    def _train_epoch(self, epoch):
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
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
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

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
