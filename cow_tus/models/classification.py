"""
"""
import logging

import torch
import torch.nn as nn
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from tqdm import tqdm

from cow_tus.models.base import BaseModel
from cow_tus.analysis.metrics import Metrics
import cow_tus.models.modules.encoders as encoders
import cow_tus.models.modules.decoders as decoders
import cow_tus.models.modules.losses as losses
from cow_tus.util import place_on_gpu, get_batch_size


class ClassificationModel(BaseModel):
    """
    """

    def __init__(self, cuda=False, devices=[0], pretrained_configs=[],
                 encoder_config=None, decoder_config=None,
                 loss_config=None, optim_config="Adam", scheduler_config=None,
                 task_configs=[]):
        """
        """
        super().__init__(cuda, devices)

        encoder_class = encoder_config["class"]
        encoder_args = encoder_config["args"]
        self.encoder = getattr(encoders, encoder_class)(**encoder_args)
        if self.cuda:
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.devices)

        decoder_class = decoder_config["class"]
        decoder_args = decoder_config["args"]
        self.decoder = getattr(decoders, decoder_class)(task_configs, **decoder_args)

        loss_class = loss_config["class"]
        loss_args = loss_config["args"]
        self.loss_fn = getattr(losses, loss_class)(**loss_args)

        self._post_init(optim_config, scheduler_config, pretrained_configs)

    def forward(self, inputs, targets):
        """
        Args:
            inputs  (torch.Tensor) a (batch_size, ...) shaped input tensor
            targets (dict) a dictionary mapping task names to targets.
        Return:
            outputs (dict) a dict that matches task names to targets.
        """
        encoding = self.encoder(inputs)
        outputs = self.decoder(encoding)
        assert(outputs.keys() == targets.keys())
        return outputs

    def calculate_loss(self, inputs, targets):
        """
        """
        return self.loss_fn(inputs=inputs, targets=targets)

    def predict(self, inputs):
        """
        """
        encoding = self._encode(inputs)
        probs = self._predict_tasks(encoding)
        return probs

    def _encode(self, inputs):
        """Accepts time series data and outputs an encoding"""
        encoding = self.encoder(inputs)
        return encoding

    def _predict_tasks(self, encoding):
        """
        """
        outputs = self.decoder(encoding)
        return probs