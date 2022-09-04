# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import torch

class Config:
    def __init__(
            self,
            num_epochs: int=40,
            learning_rate: float=3e-5,
            batch_size: int=8,
            max_sequence_length: int=256,
            log_step: int=50,
            eval_step: int=100,
            weight_decay: float=0.01,
            adam_epsilon: float=1e-8,
            max_grad_norm: float=1.0,
            inference_beta: float=0.9,
            warm_up: float=0.1,
            device: str=None,
            model_mode: str = 'train',
            # opinion_alpha: float = 1,
            preference_alpha: float = 1,
            using_active_learning = True 
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.log_step = log_step
        self.eval_step = eval_step
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.inference_beta = inference_beta
        self.warm_up = warm_up
        self.model_mode = model_mode
        self.using_active_learning = using_active_learning
        # self.opinion_alpha = opinion_alpha
        self.preference_alpha = preference_alpha
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, attr):
        return getattr(self, attr)

    def __str__(self):
        return str(self.__dict__)
