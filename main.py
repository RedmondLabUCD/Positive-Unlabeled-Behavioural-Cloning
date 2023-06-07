#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:00:42 2022

@author: qiang
"""
import argparse
import utils
import trainer
import buffer
import d3rlpy
import warnings
warnings.filterwarnings("ignore")

def main_filter(args):
    args = utils.directory_handler(args)
    buff = buffer.PUBuffer(args)
    classifier = trainer.PUTrainer(buff.args)
    if args.load_trained_filter:
        classifier.load(buff.classifier_validate)
        buff.update_pos()
    else:
        buff.set_seed_positive()
        for turn in range(args.iterations):
            classifier.train(buff.torch_loader, args.epochs_per_iteration, turn, buff.classifier_validate)
            buff.update_pos()
    if args.train_policy:
        policy_dataset = buff.init_torch_loader_to_train_policy()
        mdpd_dataset = d3rlpy.dataset.MDPDataset(
                                                observations=policy_dataset['observations'],
                                                actions=policy_dataset['actions'],
                                                rewards=policy_dataset['rewards'],
                                                terminals=policy_dataset['timeouts'],
                                            )
        algo = utils.policy_handler(buff.args)
        algo.build_with_dataset(mdpd_dataset)
        algo.fit(mdpd_dataset, n_epochs=200)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dataset-path', default='/home/qiang/Novel data recognizor/rrc_real_lift_mixed/raw_dataset.npy', help='Path to the raw mixed dataset')
    parser.add_argument('--pos-seed-dataset-path', default='/home/qiang/Novel data recognizor/rrc_real_lift_mixed/anchor_dataset.npy', help='Path to the seed-positive dataset')
    parser.add_argument('--exp-name', default="pubc", help='Experiment name')
    parser.add_argument('--models-in-ensemble', default=3, help='Number of unit models in the ensemble model')
    parser.add_argument('--ensemble-method', default='vote',  help="'avg' or 'vote'")
    parser.add_argument('--iterations', default=4, help='Iteration number for each classifier training trail')
    parser.add_argument('--negative-sampler', default='part', help="'part' or 'full'")
    parser.add_argument('--epochs-per-iteration', default=20, help='Number of epoch lasts per classifier training iteration')
    parser.add_argument('--pos-seed-rate', default=0.004, help='The top rewarding trajectaries are selected as the seed-positive dataset if it is not given')
    parser.add_argument('--th-bins', default=100, help='Adaptive threshold')
    parser.add_argument('--th-high-bound', default=0.96, help='Adaptive threshold')
    parser.add_argument('--th-fit-pow', default=10, help='Polynomial order for adaptive threshold')
    parser.add_argument('--seeds', default=0, help='random seed')
    parser.add_argument('--use-gpu', default=1, help='Use GPU for accelerating or not')
    parser.add_argument('--batch-size', default=1024, help='Classifier training batch size')
    parser.add_argument('--learning-rate', default=0.001, help='Classifier training learning rate')
    parser.add_argument('--save-path', default=None, help="Path to save the output results. If it is not given, one folder named 'save' will be created under the root project path for saving the results")
    parser.add_argument('--train-policy', default=True, help='Whether train policy')
    parser.add_argument('--policy', default='bc', help='Train which policy')
    parser.add_argument('--load-trained-filter', default=False, help='Whether use the trained filter or train from scratch')
    parser.add_argument('--trained-filter-path', default='/home/qiang/Novel data recognizor/pubc_code/save/pubc/output examples/pubc_real_lift_mixed', help='The path to the filter saved folder RATHER THAN one file path')
    parser.add_argument('--ckpt-iterations', default=4, help='How many iterations you run in the exist filter')
    args = parser.parse_args()
    main_filter(args)
