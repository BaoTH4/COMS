# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import numpy as np
from typing import Any
from torch.utils.data import Dataset, DataLoader


class DualSample(object):
    def __init__(
            self,
            original_sample: str=None,
            text: str=None,
            preference_query: list=None,
            preference_answer: list=None
    ):
        self.original_sample = original_sample
        self.text = text
        self.preference_query = preference_query
        self.preference_answer = preference_answer


class OriginalPreferenceDataset(Dataset):
    def __init__(self, pre_data):
        # self._opinion_query = pre_data.get('_opinion_query', None)  # [max_aspect_num, max_opinion_query_length]
        # self._opinion_answer = pre_data.get('_opinion_answer', None)
        # self._opinion_query_mask = pre_data.get('_opinion_query_mask', None)
        # self._opinion_query_seg =  pre_data.get('_opinion_query_seg', None)
        # self._opinion_answer_mask = pre_data.get('_opinion_answer_mask', None)
        self._preference_query = pre_data.get('_preference_query', None)
        self._preference_answer = pre_data.get('_preference_answer', None)
        self._preference_query_mask = pre_data.get('_preference_query_mask', None)
        self._preference_query_seg = pre_data.get('_preference_query_seg', None)

    def __str__(self):
        str = '----------------------------------------\n'
        # str += f"_opinion_query: {self._opinion_query[0]}\n"
        # str += f"_opinion_answer: {self._opinion_answer[0]}\n"
        # str += f"_opinion_query_mask: {self._opinion_query_mask[0]}\n"
        # str += f"_opinion_query_seg: {self._opinion_query_seg[0]}\n"
        # str += f"_opinion_answer_mask: {self._opinion_answer_mask[0]}\n"
        str += f"_preference_query: {self._preference_query[0]}\n"
        str += f"_preference_answer: {self._preference_answer[0]}\n"
        str += f"_preference_query_mask: {self._preference_query_mask[0]}\n"
        str += f"_preference_query_seg: {self._preference_query_seg[0]}\n"
        str += '----------------------------------------\n'

        return str


class MRCCPCDataset(Dataset):
    def __init__(self, data=None):
        # self._opinion_query = data._opinion_query
        # self._opinion_answer = data._opinion_answer
        # self._opinion_query_mask = data._opinion_query_mask
        # self._opinion_query_seg = data._opinion_query_seg
        # self._opinion_answer_mask = data._opinion_answer_mask
        self._preference_query = data._preference_query
        self._preference_answer = data._preference_answer
        self._preference_query_mask = data._preference_query_mask
        self._preference_query_seg = data._preference_query_seg

    def get_batch_num(self, batch_size):
        return len(self._preference_query) // batch_size

    def __len__(self):
        return len(self._preference_query)

    def __getitem__(self, item):
        #TODO: OPinion and preference
        # opinion_query = self._opinion_query[item]
        # opinion_answer = self._opinion_answer[item]
        # opinion_query_mask = self._opinion_query_mask[item]
        # opinion_query_seg = self._opinion_query_seg[item]
        # opinion_answer_mask = self._opinion_answer_mask[item]
        preference_query = self._preference_query[item]
        preference_answer = self._preference_answer[item]
        preference_query_mask = self._preference_query_mask[item]
        preference_query_seg = self._preference_query_seg[item]

        return {
            # "opinion_query": np.array(opinion_query),
            # "opinion_answer": np.array(opinion_answer),
            # "opinion_query_mask": np.array(opinion_query_mask),
            # "opinion_query_seg": np.array(opinion_query_seg),
            # "opinion_answer_mask": np.array(opinion_answer_mask),
            "preference_query": np.array(preference_query),
            "preference_answer": np.array(preference_answer),
            "preference_query_mask": np.array(preference_query_mask),
            "preference_query_seg": np.array(preference_query_seg)
        }


def generate_fi_batches(dataset, batch_size, shuffle=True, drop_last=True, device=None):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            out_dict[name] = data_dict[name].to(device)

        yield out_dict