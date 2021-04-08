import torch
import torchreid
import os
from collections import OrderedDict
import time
import os.path as osp
import sys
import logging

from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

log_name = 'test_log' 
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join('base_', log_name))


checkpoint_path1 = 'models/market/model_name.tar'

def load_state_dict(checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path) 
        state_dict_key = 'state_dict'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint1(model, checkpoint_path1, strict=True):
    state_dict = load_state_dict(checkpoint_path1)
    model.load_state_dict(state_dict, strict=strict)


_logger = logging.getLogger(__name__)

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501', 
    height=224, 
    width=224, 
    batch_size_train=128,
    batch_size_test=100,
)

train_loader = datamanager.train_loader
test_loader = datamanager.test_loader


model = torchreid.models.create_model(
    model_name= 'uv_p16',
    pretrained=False, 
    num_classes=datamanager.num_train_pids,
    in_chans=3,
)


model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='radam',
    lr=0.001
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[10,20]
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True,
)

start_epoch = load_checkpoint1( model, checkpoint_path1)


engine.run(
    test_only=True,
)




