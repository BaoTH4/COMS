# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from compar.comparative_preference import utils
from torch.nn import functional as F
from compar.comparative_preference.config import Config
from transformers import get_linear_schedule_with_warmup, AdamW
from compar.comparative_preference.dataset import (
    OriginalPreferenceDataset,
    MRCCPCDataset,
    generate_fi_batches
)
from compar.constants import ID2PREFERENCE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s",
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)


class Trainer(object):
    def __init__(
            self,
            config: Config=None,
            model: Any=None,
            train_data: OriginalPreferenceDataset=None,
            test_data: OriginalPreferenceDataset=None,
            test_standard: Any=None,
            tokenizer: Any=None,
            **kwargs
    ):
        self.config = config
        self.model = model.to(self.config.device)
        self.train_dataset = MRCCPCDataset(train_data)
        self.test_dataset = MRCCPCDataset(test_data)
        self.test_standard = test_standard
        self.tokenizer = tokenizer

    def train(
            self,
            model_dir: str='./models/subject_extraction',
            model_name: str='best_model',
            **kwargs
    ):
        logger.info(f"Device: {self.config.device}")
        logger.info(f'➖➖➖➖➖➖➖ Dataset Info ➖➖➖➖➖➖➖')
        logger.info(f'Length of train dataset: {len(self.train_dataset)}')
        logger.info(f'Length of test dataset : {len(self.test_dataset)}')
        logger.info(f'➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖')

        batch_num_train = self.train_dataset.get_batch_num(self.config.batch_size)

        # Optimizer
        logger.info('Initial optimizer...')
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon, correct_bias=False)

        start_epoch = 1
        logger.info('New model and optimizer from epoch 0')

        # Scheduler
        training_steps = self.config.num_epochs * batch_num_train
        warmup_steps = int(training_steps * self.config.warm_up)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

        logger.info(f"Begin training Subject Extraction task...")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        best_test_f1 = 0.

        if self.config.using_active_learning:
            for epoch in tqdm(range(start_epoch, self.config.num_epochs + 1)):
                self.model.train()
                self.model.zero_grad()

                if epoch == 1:
                    batch_generator = generate_fi_batches(
                        self.train_dataset, batch_size=self.config.batch_size, device=self.config.device)
                else:
                    batch_generator = generate_fi_batches(
                        next_data, batch_size=self.config.batch_size, device=self.config.device, shuffle=False)
                    batch_num_train = next_data.get_batch_num(self.config.batch_size)

                for batch_index, batch_dict in enumerate(batch_generator):
                    optimizer.zero_grad()

                    preference_socres = self.model(
                        batch_dict['preference_query'].view(-1, batch_dict['preference_query'].size(-1)),
                        batch_dict['preference_query_mask'].view(-1, batch_dict['preference_query_mask'].size(-1)),
                        batch_dict['preference_query_seg'].view(-1, batch_dict['preference_query_seg'].size(-1))
                    )

                    lossP = utils.calculate_preference_loss_2(preference_socres, batch_dict['preference_answer'])

                    loss_sum = self.config.preference_alpha*lossP

                    loss_sum.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
             
                    # train logger
                    if batch_index % self.config.log_step == 0:
                        logger.info(
                            'Epoch {}/{} - Batch {}/{}:\t Loss:{}'.
                                format(epoch, self.config.num_epochs, batch_index, batch_num_train, round(loss_sum.item(), 4))
                        )

                cur_batch_generator_train = generate_fi_batches(
                    dataset=self.train_dataset, batch_size=1, shuffle=False, device=self.config.device)

                next_data = self.generate_next_data(cur_batch_generator_train, threshold_score=0.97) ##MRCCPCDataset

                # validation
                batch_generator_test = generate_fi_batches(
                    dataset=self.test_dataset, batch_size=1, shuffle=False, device=self.config.device)

                logger.info(f"Evaluate Test...")
                f1 = self.evaluate(batch_generator_test)
                if f1 > best_test_f1:
                    # TODO: Storage model
                    best_test_f1 = f1
                    model_file_path = os.path.join(model_dir, f"{model_name}.pt")

                    state = {'net': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, model_file_path)
                    logger.info(f"📥 Saved best test model to {model_file_path}.")

        else:
            for epoch in tqdm(range(start_epoch, self.config.num_epochs + 1)):
                self.model.train()
                self.model.zero_grad()

                batch_generator = generate_fi_batches(
                    self.train_dataset, batch_size=self.config.batch_size, device=self.config.device)

                for batch_index, batch_dict in enumerate(batch_generator):
                    optimizer.zero_grad()

                    preference_socres = self.model(
                        batch_dict['preference_query'].view(-1, batch_dict['preference_query'].size(-1)),
                        batch_dict['preference_query_mask'].view(-1, batch_dict['preference_query_mask'].size(-1)),
                        batch_dict['preference_query_seg'].view(-1, batch_dict['preference_query_seg'].size(-1))
                    )

                    lossP = utils.calculate_preference_loss_2(preference_socres, batch_dict['preference_answer'])

                    loss_sum = self.config.preference_alpha*lossP

                    loss_sum.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
             
                    # train logger
                    if batch_index % self.config.log_step == 0:
                        logger.info(
                            'Epoch {}/{} - Batch {}/{}:\t Loss:{}'.
                                format(epoch, self.config.num_epochs, batch_index, batch_num_train, round(loss_sum.item(), 4))
                        )

                # validation
                batch_generator_test = generate_fi_batches(
                    dataset=self.test_dataset, batch_size=1, shuffle=False, device=self.config.device)

                logger.info(f"Evaluate Test...")
                f1 = self.evaluate(batch_generator_test)
                if f1 > best_test_f1:
                    # TODO: Storage model
                    best_test_f1 = f1
                    model_file_path = os.path.join(model_dir, f"{model_name}.pt")

                    state = {'net': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, model_file_path)
                    logger.info(f"📥 Saved best test model to {model_file_path}.")
   
    def generate_next_data(self, cur_batch_generator, score_method = 'confidence', threshold_score=0.9):
        self.model.eval()
        _preference_query = []
        _preference_answer = []
        _preference_query_mask = []
        _preference_query_seg = []
        score_list = []
        
        for batch_index, batch_dict in enumerate(cur_batch_generator):
            preference_logits = self.model(
                batch_dict['preference_query'].view(-1, batch_dict['preference_query'].size(-1)),
                batch_dict['preference_query_mask'].view(-1, batch_dict['preference_query_mask'].size(-1)),
                batch_dict['preference_query_seg'].view(-1, batch_dict['preference_query_seg'].size(-1))
            )

            if score_method == 'confidence':
                pref_scores = F.softmax(preference_logits, dim = -1)
                confidence_score, _ = torch.max(pref_scores, dim = -1)
                score_list.append(confidence_score.item())

            _preference_query.append(batch_dict['preference_query'].view(-1, batch_dict['preference_query'].size(-1)).tolist()[0])
            _preference_query_mask.append(batch_dict['preference_query_mask'].view(-1, batch_dict['preference_query_mask'].size(-1)).tolist()[0])
            _preference_query_seg.append(batch_dict['preference_query_seg'].view(-1, batch_dict['preference_query_seg'].size(-1)).tolist()[0])
            _preference_answer.append(batch_dict['preference_answer'].tolist()[0])

        ##Sort threshold scores by ascending order
        score_list = torch.tensor(score_list)
        _, indices = torch.sort(score_list)
        
        ##Chooose bad samples by confidence and threshold score
        bad_sample_indices = []
        _bad_query = []
        _bad_mask = []
        _bad_seg = []
        _bad_answer = []

        for indice in indices:
            if score_list[indice.item()].item() < threshold_score:
                bad_sample_indices.append(indice.item())

        if bad_sample_indices != []:
            bad_sample_indices = sorted(bad_sample_indices,reverse=True)        
            for indice in bad_sample_indices:
                bad_query = _preference_query.pop(indice)
                bad_mask = _preference_query_mask.pop(indice)
                bad_seg = _preference_query_seg.pop(indice)
                bad_answer = _preference_answer.pop(indice)

                _bad_query.append(bad_query)
                _bad_query.append(bad_query)
                _bad_mask.append(bad_mask)
                _bad_mask.append(bad_mask)
                _bad_seg.append(bad_seg)
                _bad_seg.append(bad_seg)
                _bad_answer.append(bad_answer)
                _bad_answer.append(bad_answer)

            bad_sample_dict = {
                'query': _bad_query,
                'mask': _bad_mask,
                'seg': _bad_seg,
                'answer': _bad_answer
            }

            bad_sample_df = pd.DataFrame(data=bad_sample_dict, columns=['query', 'mask', 'seg', 'answer'])
            bad_sample_df = bad_sample_df.sample(frac=1)

            _bad_query = bad_sample_df['query'].tolist()
            _bad_mask = bad_sample_df['mask'].tolist()
            _bad_seg = bad_sample_df['seg'].tolist()
            _bad_answer = bad_sample_df['answer'].tolist()

            _preference_query = _bad_query + _preference_query
            _preference_query_mask = _bad_mask + _preference_query_mask
            _preference_query_seg = _bad_seg + _preference_query_seg
            _preference_answer = _bad_answer + _preference_answer

            _preference_query = torch.tensor(_preference_query)
            _preference_query_mask = torch.tensor(_preference_query_mask)
            _preference_query_seg = torch.tensor(_preference_query_seg)
            _preference_answer = torch.tensor(_preference_answer)

        else:
            _preference_query = torch.tensor(_preference_query)
            _preference_query_mask = torch.tensor(_preference_query_mask)
            _preference_query_seg = torch.tensor(_preference_query_seg)
            _preference_answer = torch.tensor(_preference_answer)

            _preference_query = _preference_query[indices, :]
            _preference_query_mask = _preference_query_mask[indices, :]
            _preference_query_seg = _preference_query_seg[indices, :]
            _preference_answer = _preference_answer[indices, :]

        ori_data_dict = {
            "_preference_query": _preference_query.tolist(),
            "_preference_answer": _preference_answer.tolist(),
            "_preference_query_mask": _preference_query_mask.tolist(),
            "_preference_query_seg": _preference_query_seg.tolist()
        }

        ori_data = OriginalPreferenceDataset(ori_data_dict)

        return MRCCPCDataset(ori_data)

    def get_end_index(self, answer_mask, id):
        try:
            if id + 1 < answer_mask.size(1):
                while (answer_mask[0][id + 1] == 0):
                    id += 1
                    if id + 1 >= len(answer_mask):
                        return id
                return id
            else:
                return id
        except:
            logger.info(f"answer_mask = {answer_mask.size()} {answer_mask.size(1)}; id = {id}")

    def evaluate(self, test_batch_generator):
        self.model.eval()
        pref_predicts = []
        pref_targets = []
        public_ids = []
        
        for batch_index, batch_dict in enumerate(test_batch_generator):
            pref_target = self.test_standard['pref_text'][batch_index]
            pref_targets.append(pref_target)
            ##public id
            public_id = self.test_standard['public_ids'][batch_index]
            public_ids.append(public_id)
            preference_logits = self.model(
                batch_dict['preference_query'].view(-1, batch_dict['preference_query'].size(-1)),
                batch_dict['preference_query_mask'].view(-1, batch_dict['preference_query_mask'].size(-1)),
                batch_dict['preference_query_seg'].view(-1, batch_dict['preference_query_seg'].size(-1))
            )
            pref_scores = F.softmax(preference_logits, dim = -1)
            _, pref_predict = torch.max(pref_scores, dim = -1)
            pref_predicts.append(ID2PREFERENCE[pref_predict.item()])

        assert len(pref_targets) == len(pref_predicts)
        acc = accuracy_score(y_true=pref_targets, y_pred=pref_predicts)
        precision = precision_score(y_true=pref_targets, y_pred=pref_predicts, average='weighted')
        recall = recall_score(y_true=pref_targets, y_pred=pref_predicts, average='weighted')
        f1 = f1_score(y_true=pref_targets, y_pred=pref_predicts, average='weighted')

        logger.info(
            'Comparative Preference - Accuracy: {} ; Precision: {} ; Recall: {} ; F1: {}'
                .format(acc, precision, recall, f1)
        )

        logger.info(
            'Comparative Preference classification report:\n{}'
            .format(
                classification_report(
                    pref_targets,
                    pref_predicts,
                    labels=list(ID2PREFERENCE.values()),
                    digits=4
                )
            )
        )

        cm = confusion_matrix(pref_targets, pref_predicts, labels=list(ID2PREFERENCE.values()))
        logger.info('Confusion matrix:\n{}'.format(cm))
        
        return f1