import torch
import inspect
import json
import yaml
import math
import os
import sys

from general_utils import log

import numpy as np
from functools import partial
from os.path import expanduser, join, isfile, basename

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
from torch.utils.data import DataLoader

from general_utils import TrainingLogger, get_attribute, filter_args, log, training_config_from_cli_args

def cosine_warmup_lr(i, warmup=10, max_iter=90):
    """ Cosine LR with Warmup """
    if i < warmup:
        return (i+1)/(warmup+1)
    else:
        return 0.5 + 0.5*math.cos(math.pi*(((i-warmup)/(max_iter- warmup))))

def validate(model, dataset, config):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    metric_class, use_metric = config.val_metric_class, config.use_val_metric
    loss_fn = get_attribute(config.loss)

    model.eval()
    model.cuda()

    if metric_class is not None:
        metric = get_attribute(metric_class)()

    with torch.no_grad():

        i, losses = 0, []
        for data_x, data_y in data_loader:

            data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x]
            data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

            prompts = model.sample_prompts(data_x[1], prompt_list=('a photo of a {}',))
            pred, visual_q, _, _  = model(data_x[0], prompts, return_features=True)

            if metric_class is not None:
                metric.add([pred], data_y)

            # pred = model(data_x[0], prompts)
            # loss = loss_fn(pred[0], data_y[0])
            loss = loss_fn(pred, data_y[0])
            losses += [float(loss)]

            i += 1

            if config.val_max_iterations is not None and i > config.val_max_iterations:
                break

    if use_metric is None:
        return np.mean(losses), {}, False
    else:
        metric_scores = {m: s for m, s in zip(metric.names(), metric.value())} if metric is not None else {}
        return np.mean(losses), metric_scores, True
