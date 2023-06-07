#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:20:44 2023

@author: qiang
"""
import pandas as pd
import random
import os
import numpy as np
import torch
from typing_extensions import Protocol
from typing import Any, Dict, Iterator, List, Optional
import structlog
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import json
from contextlib import contextmanager
import scipy.signal as sg
import matplotlib.pyplot as plt
from copy import copy
from d3rlpy.algos import PLAS,IQL,TD3PlusBC,BC,CRR


RRC_LIFT_TASK_OBS_DIM = 139
RRC_PUSH_TASK_OBS_DIM = 97
RRC_ACTION_DIM = 9
RRC_MAX_ACTION = 0.397

def device_handler(args):
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
    args.device = device
    return args

def directory_handler(args):
    if not args.save_path:
        proj_root_path = os.path.split(os.path.realpath(__file__))[0]
        args.save_path = f'{proj_root_path}/save'
    if os.path.split(args.save_path)[-1] != args.exp_name:
        args.save_path = f'{args.save_path}/{args.exp_name}'
    return args

def policy_handler(args):
    if args.policy == 'bc':
        algorithm = BC(learning_rate=0.001, 
                       batch_size=1024,
                       use_gpu=args.use_gpu)
    elif args.policy == 'td3bc':
        algorithm = TD3PlusBC(actor_learning_rate=0.0003,
                        critic_learning_rate=0.0003,
                        batch_size=256,
                        use_gpu=args.use_gpu)
    elif args.policy == 'iql':
        algorithm = IQL(actor_learning_rate=0.0003,
                        critic_learning_rate=0.0003,
                        batch_size=256,
                        expectile=0.7,
                        weight_temp=3.0,
                        use_gpu=args.use_gpu)
    elif args.policy == 'crr':
        algorithm = CRR(actor_learning_rate=0.0003,
                        critic_learning_rate=0.0003,
                        batch_size=256,
                        beta=1.0,
                        use_gpu=args.use_gpu)
    elif args.policy == 'plas':
        algorithm = PLAS(actor_learning_rate=0.0001,
                        critic_learning_rate=0.001,
                        warmup_steps=500000,
                        beta=0.5,
                        use_gpu=args.use_gpu)
    return algorithm


def adap_probs(probs, bins=100, fit_pow=8, prob_th=0.96, plot=True, save_plot_path = None):
    def get_minima(values: np.ndarray):
        min_index = sg.argrelmin(values)[0]
        return min_index, values[min_index]

    def ydata_gen(probs):
        return np.histogram(probs,bins=bins)[0]
    
    unit = (probs.max() - probs.min()) / bins
    xdata = []
    for i in range(bins):
        xdata.append(probs.min() + (i * unit))
    xdata = np.array(xdata)
    ydata = np.array(ydata_gen(probs))
    print(probs.max())
    print(probs.min())
    
    coeffi = np.polyfit(xdata, ydata, fit_pow)
    pln = np.poly1d(coeffi)
    y_pred=pln(xdata)

    idxs, values = get_minima(y_pred)

    if idxs.shape[0] > 1:
        if xdata[idxs[-1]] >= prob_th:
            adap_p = xdata[idxs[-2]]
            adap_v = values[-2]
        else:
            adap_p = xdata[idxs[-1]]
            adap_v = values[-1]
    else:
        if xdata[idxs[-1]] >= prob_th:
            adap_p = prob_th
            adap_v = None
        else:
            adap_p = xdata[idxs[-1]]
            adap_v = values[-1]

    plt.figure()
    
    plt.plot(xdata, ydata, '*',color='gold',label='original values')
    plt.plot(xdata, y_pred, color='r',label='polyfit values')
    plt.scatter(adap_p, adap_v, s=40,marker='D',color='b', label='adapted point')
    plt.legend(loc=0)
    if plot:
        plt.show()
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=600,bbox_inches='tight')
    print(f'The adaptive probability is {adap_p}')
    return adap_p

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pos_seed_by_reward(dataset, percent=0.02):
    avg_episodic_rewards = []
    temp_rewards = []
    count = 0
    for idx, timeout in enumerate(dataset['timeouts']):
        temp_rewards.append(dataset['rewards'][idx])
        if timeout:
            avg = np.array(copy(temp_rewards)).mean()
            avg_episodic_rewards.append(avg)
            temp_rewards = []
            count += 1
    pos_seed_num = int(percent * count)
    avg_episodic_rewards.sort()
    reward_pos_seed = avg_episodic_rewards[-pos_seed_num]
    
    pos_seed_obs = []
    pos_seed_actions = []
    temp_rewards = []
    temp_actions = []
    temp_obs = []
    num_selected = 0
    for idx, timeout in enumerate(dataset['timeouts']):
        temp_rewards.append(dataset['rewards'][idx])
        temp_obs.append(dataset['observations'][idx].tolist())
        temp_actions.append(dataset['actions'][idx].tolist())
        if timeout:
            avg = np.array(copy(temp_rewards)).mean()
            if avg >= reward_pos_seed:
                num_selected += 1
                pos_seed_obs += temp_obs
                pos_seed_actions += temp_actions
            temp_obs = []
            temp_actions = []
            temp_rewards = []
    
    print(f'{num_selected}/{count} pos_seeds are selected')
    temp_dataset = {}
    temp_dataset['observations'] = np.array(pos_seed_obs)
    temp_dataset['actions'] = np.array(pos_seed_actions)
    return temp_dataset

class matrix():
    def __init__(self,
                 ):
        self.title = ['TURN','ACC']
        self.data = []
        
    def update(self,log):
        self.data.append(log)
    
    def save(self,path):
        dataframe = pd.DataFrame(data=self.data,columns=self.title)
        dataframe.to_csv(path,index=False,sep=',')
        
    def clear(self):
        self.data=[]
        
class _SaveProtocol(Protocol):
    def save_model(self, fname: str) -> None:
        ...

# default json encoder for numpy objects


def default_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise ValueError(f"invalid object type: {type(obj)}")


LOG: structlog.BoundLogger = structlog.get_logger(__name__)


class TrainLogger:

    _experiment_name: str
    _logdir: str
    _save_metrics: bool
    _verbose: bool
    _metrics_buffer: Dict[str, List[float]]
    _params: Optional[Dict[str, float]]
    _writer: Optional[SummaryWriter]

    def __init__(
        self,
        experiment_name: str,
        tensorboard_dir: Optional[str] = None,
        save_metrics: bool = True,
        root_dir: str = "logs",
        verbose: bool = True,
        with_timestamp: bool = True,
    ):
        self._save_metrics = save_metrics
        self._verbose = verbose

        # add timestamp to prevent unintentional overwrites
        while True:
            if with_timestamp:
                date = datetime.now().strftime("%Y%m%d%H%M%S")
                self._experiment_name = experiment_name + "_" + date
            else:
                self._experiment_name = experiment_name

            if self._save_metrics:
                self._logdir = os.path.join(root_dir, self._experiment_name)
                if not os.path.exists(self._logdir):
                    os.makedirs(self._logdir)
                    LOG.info(f"Directory is created at {self._logdir}")
                    break
                if with_timestamp:
                    time.sleep(1.0)
                if os.path.exists(self._logdir):
                    LOG.warning(
                        f"You are saving another logger into {self._logdir}, this may cause unintentional overite")
                    break
            else:
                break

        self._metrics_buffer = {}

        if tensorboard_dir:
            tfboard_path = self._logdir
            self._writer = SummaryWriter(logdir=tfboard_path)
        else:
            self._writer = None

        self._params = None

    def add_params(self, params: Dict[str, Any]) -> None:
        assert self._params is None, "add_params can be called only once."

        if self._save_metrics:
            # save dictionary as json file
            params_path = os.path.join(self._logdir, "params.json")
            with open(params_path, "w") as f:
                json_str = json.dumps(
                    params, default=default_json_encoder, indent=2
                )
                f.write(json_str)

            if self._verbose:
                LOG.info(
                    f"Parameters are saved to {params_path}", params=params
                )
        elif self._verbose:
            LOG.info("Parameters", params=params)

        # remove non-scaler values for HParams
        self._params = {k: v for k, v in params.items() if np.isscalar(v)}

    def add_metric(self, name: str, value: float) -> None:
        if name not in self._metrics_buffer:
            self._metrics_buffer[name] = []
        self._metrics_buffer[name].append(value)

    def commit(self, epoch: int, step: int) -> Dict[str, float]:
        metrics = {}
        for name, buffer in self._metrics_buffer.items():

            metric = sum(buffer) / len(buffer)

            if self._save_metrics:
                path = os.path.join(self._logdir, f"{name}.csv")
                with open(path, "a") as f:
                    print(f"{epoch},{step},{metric}", file=f)

                if self._writer:
                    self._writer.add_scalar(f"metrics/{name}", metric, epoch)

            metrics[name] = metric

        if self._verbose:
            LOG.info(
                f"{self._experiment_name}: epoch={epoch} step={step}",
                epoch=epoch,
                step=step,
                metrics=metrics,
            )

        if self._params and self._writer:
            self._writer.add_hparams(
                self._params,
                metrics,
                name=self._experiment_name,
                global_step=epoch,
            )

        # initialize metrics buffer
        self._metrics_buffer = {}
        return metrics

    def save_model(self, epoch: int, algo: _SaveProtocol) -> None:
        if self._save_metrics:
            # save entire model
            model_path = os.path.join(self._logdir, f"model_{epoch}.pt")
            algo.save_model(model_path)
            LOG.info(f"Model parameters are saved to {model_path}")

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    @contextmanager
    def measure_time(self, name: str) -> Iterator[None]:
        name = "time_" + name
        start = time.time()
        try:
            yield
        finally:
            self.add_metric(name, time.time() - start)

    @property
    def logdir(self) -> str:
        return self._logdir

    @property
    def experiment_name(self) -> str:
        return self._experiment_name