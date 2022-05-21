# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import logging
import argparse
import math
import os
import random
import time

import numpy

from sklearn import metrics
import datetime
from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('加载Bert...')
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            print('Bert加载完毕.')
            print(f'>>> 使用设备:{opt.device} 训练.')
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            print('加载预训练向量...')
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            print('预训练向量加载完毕.')
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        print(f'> training dataset count: {len(self.trainset.data)}.')
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        print(f'> testing dataset count: {len(self.testset.data)}.')
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        best_info = '>>> val_acc: 0.0, val_precision: 0.0 val_recall: 0.0, val_f1: 0.0'
        for i_epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('>>> epoch: {}.'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    print('E2E-ABSA >>>', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_prec, val_rec, val_f1 = self._evaluate_acc_f1(val_data_loader)
            print('E2E-ABSA >>>', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print('>>> val_acc: {:.4f}, val_precision: {:.4f} val_recall: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_prec, val_rec, val_f1))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                best_info = '>>> val_acc: {:.4f}, val_precision: {:.4f} val_recall: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_prec, val_rec, val_f1)
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_f1_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                print('>> saved: {}'.format(path))
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('E2E-ABSA >>>', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                print('>>> early stop.')
                print(f'BEST PERFORMANCE(模型最佳表现): {best_info}')
                break
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        x, y = t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu()
        precision, recall, f1_score, = metrics.precision_score(x, y, labels=[0, 1, 2], average='micro'),\
        metrics.recall_score(x, y, labels=[0, 1, 2], average='micro'),metrics.f1_score(x, y, labels=[0, 1, 2], average='micro')
        return acc, precision, recall, f1_score

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        print(f'you can download the best model from {best_model_path}')
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_prec, test_recall, test_f1 = self._evaluate_acc_f1(test_data_loader)
        print('>>> test_acc: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_prec, test_recall, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str, choices=['lstm','td_lstm','tc_lstm','atae_lstm'
        ,'ian','memnet','ram','cabasc','tnet_lf','aoa,mgan','bert_spc','aen_bert','lcf_bert'], help=
    'choose model from lstm,td_lstm,tc_lstm,atae_lstm,ian,memnet,ram,cabasc,tnet_lf,aoa,mgan,bert_spc,aen_bert,lcf_bert')
    parser.add_argument('--dataset', default='twitter', choices=['twitter', 'acl14shortdata', 'SemEval2014',
                                                                 'SemEval2015', 'SemEval2016', 'twitter_know',
                                                                 'acl14shortdata_know', 'SemEval2014_know',
                                                                 'SemEval2015_know', 'SemEval2016_know'], type=str,
                        help='choose from twitter, acl14shortdata, SemEval2014, SemEval2015, SemEval2016 |||_know')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=100, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        # 'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'acl14shortdata': {
            'train': './datasets/acl14shortdata/train.tsv',
            'test': './datasets/acl14shortdata/dev.tsv'
        },
        'acl14shortdata_know': {
            'train': './datasets/acl14shortdata/output_know/train.tsv',
            'test': './datasets/acl14shortdata/output_know/dev.tsv'
        },
        'SemEval2014': {
            'train': './datasets/laprest14/train.tsv',
            'test': './datasets/laprest14/dev.tsv'
        },
        'SemEval2014_know': {
            'train': './datasets/laprest14/output_know/train.tsv',
            'test': './datasets/laprest14/output_know/dev.tsv'
        },
        'SemEval2015': {
            'train': './datasets/rest15/train.tsv',
            'test': './datasets/rest15/dev.tsv'
        },
        'SemEval2015_know': {
            'train': './datasets/rest15/output_know/train.tsv',
            'test': './datasets/rest15/output_know/dev.tsv'
        },
        'SemEval2016': {
            'train': './datasets/rest16/train.tsv',
            'test': './datasets/rest16/dev.tsv'
        },
        'SemEval2016_know': {
            'train': './datasets/rest16/output_know/train.tsv',
            'test': './datasets/rest16/output_know/dev.tsv'
        },
        'twitter': {
            'train': './datasets/twitter/train.tsv',
            'test': './datasets/twitter/dev.tsv'
        },
        'twitter_know': {
            'train': './datasets/twitter/output_know/train.tsv',
            'test': './datasets/twitter/output_know/dev.tsv'
        }
    }
    input_colses = {
        'lstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        # 'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
