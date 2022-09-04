# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import torch
from torch.nn import functional as F
from compar.constants import ID2PREFERENCE
from transformers import DebertaV2Tokenizer
from compar.comparative_preference.model import MRCForCPC


class Inference(object):
    def __init__(self, encoder_name_or_path, model_path = './models/best_model/pref-debertav3-xsmall-active_choosing-autoscale-08306.pt' ,device = 'cuda'):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(encoder_name_or_path)
        self.device = device
        self.encoder_name_or_path = encoder_name_or_path
        self.model = MRCForCPC(self.encoder_name_or_path)
        loaded_state_dict = torch.load(model_path)
        self.model.load_state_dict(loaded_state_dict['net'])
        self.model.to(self.device)

    
    def create_inputs(self, subjects, objects, aspects, sentences):
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        max_len = 0

        for j in range(len(subjects)):
            input_ids = []
            tokens_ids = []
            text_tokens_ids = []

            if subjects[j] == '':
                subjects[j] = '-'
            elif objects[j] == '':
                objects[j] = '-'
            elif aspects[j] == '':
                aspects[j] == '-'


            query = f'which comparative preference given subject {subjects[j]} , object {objects[j]} and aspect {aspects[j]} ?'
            query = query.lower().split()

            for i, word in enumerate(query):
                tokens = self.tokenizer.encode(word)
                tokens_ids.extend(tokens[1: -1])
        
            input_ids.extend([self.tokenizer.cls_token_id])
            input_ids.extend(tokens_ids)

            for i, word in enumerate(sentences[j].lower().split()):
                tokens = self.tokenizer.encode(word)
                text_tokens_ids.extend(tokens[1:-1])
        
            input_ids.extend([self.tokenizer.sep_token_id])
            input_query_seg = [0] * len(input_ids) + [1] * (len(text_tokens_ids) + 1)
            input_ids.extend(text_tokens_ids)
            input_ids.extend([self.tokenizer.sep_token_id])
            input_query_mask = [1] * len(input_ids)

            assert len(input_ids) == len(input_query_seg) == len(input_query_mask)

            if len(input_ids) > max_len:
                max_len = len(input_ids)

            input_ids_list.append(input_ids)
            attention_mask_list.append(input_query_mask)
            token_type_ids_list.append(input_query_seg)

        for i in range(len(input_ids_list)):
            pad_num = max_len - len(input_ids_list[i])

            input_ids_list[i].extend([self.tokenizer.pad_token_id] * pad_num)
            attention_mask_list[i].extend([0] * pad_num)
            token_type_ids_list[i].extend([1] * pad_num) 

        if self.device == 'cuda':
            return torch.tensor([input_ids_list]).to(self.device), torch.tensor([token_type_ids_list]).to(self.device), torch.tensor([attention_mask_list]).to(self.device)
        else:
            return torch.tensor([input_ids_list]), torch.tensor([token_type_ids_list]), torch.tensor([attention_mask_list])

    def inference_by_pytorch(self, subjects, objects, aspects, sentences):
        _input_ids, _token_type_ids, _attention_mask = self.create_inputs(subjects, objects, aspects, sentences)
        logits = self.model(_input_ids.squeeze(dim=0), _attention_mask.squeeze(dim=0), _token_type_ids.squeeze(dim=0))
        scores = F.softmax(logits, dim = -1)
        confidence_scores , predicts = torch.max(scores, dim = -1)

        results = []
        for i in range(len(subjects)):
            dict_result = {
                'class': ID2PREFERENCE[predicts[i].item()],
                'confidence': round(confidence_scores[i].item(), 4)
            }
            results.append(dict_result)
        # quadruple = self.process_output(subjects, objects, aspects, predicts)
        # return quadruple
        return results

    def process_output(self, subjects, objects, aspects, predicts):
        quadruple = []

        for i in range(len(subjects)):
            quadruplet = (subjects[i], objects[i], aspects[i], ID2PREFERENCE[predicts[i].item()])
            quadruple.append(quadruplet)
        
        return quadruple





