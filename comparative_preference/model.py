# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

from pickle import NONE
from turtle import forward
import torch
import logging
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Tokenizer
from compar.constants import ID2PREFERENCE

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s",
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)

class MRCForCPC(nn.Module):
    def __init__(self, encoder_name_or_path):
        super(MRCForCPC, self).__init__()
        self.cpc_encoder = DebertaV2Model.from_pretrained(encoder_name_or_path)
        logger.info(f"Loaded `{encoder_name_or_path}` model !")
        self.cpc_cls = nn.Linear(self.cpc_encoder.config.hidden_size, len(ID2PREFERENCE.keys()))
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        CLS_hidden_state = self.cpc_encoder(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )[0][:, 0, :]

        preference_score = self.cpc_cls(CLS_hidden_state)

        return preference_score

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        return cls(encoder_name_or_path=model_name_or_path)