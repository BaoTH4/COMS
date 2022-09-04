# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import re
import json
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
from compar.constants import PREFERENCE2ID
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from compar.comparative_preference.dataset import DualSample, OriginalPreferenceDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s",
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)

class DataParser:
    @staticmethod
    def load_data_from_json(train_paths: Union[str, list]=None, test_path: str=None):
        train_sentences = []
        test_sentences = []

        if isinstance(train_paths, str):
            with open(train_paths, 'r', encoding='utf-8') as f:
                try:
                    train_sentences = json.load(f)['document'].get('sentences', [])
                except Exception as e:
                    raise ValueError(f"{e}")
        elif isinstance(train_paths, list):
            for train_path in train_paths:
                with open(train_path, 'r', encoding='utf-8') as f:
                    try:
                        tmp_sentences = json.load(f)['document'].get('sentences', [])
                        train_sentences.extend(tmp_sentences)
                    except Exception as e:
                        raise ValueError(f"{e}")

        if test_path:
            with open(test_path, 'r', encoding='utf-8') as f:
                try:
                    test_sentences = json.load(f)['document'].get('sentences', [])
                except Exception as e:
                    raise ValueError(f"{e}")

        train_texts = []
        train_quadruple_data = []
        for sentence in tqdm(train_sentences, desc=f"Make quadruple data"):
            public_id = sentence.get('id', '')
            content = sentence.get('content', '')
            tags = sentence.get('tags', [])

            if not tags:
                logger.warning(f'--> Missing label tags in id={public_id} and content={content}')
                continue

            quadruple = DataParser.get_quadruples(public_id, content, tags)

            if not quadruple:
                logger.error(f'Quadruple empty in id={public_id} and content={content}')
                continue

            train_texts.append(content)
            train_quadruple_data.append(quadruple)

        ##Filtering to get each tag as sample
        new_train_texts = []
        new_train_quadruple_data = []
        for i in range(len(train_quadruple_data)):
            for quadruple in train_quadruple_data[i]:
                new_train_texts.append(train_texts[i])
                new_train_quadruple_data.append(quadruple)

        train_texts = new_train_texts
        train_quadruple_data = new_train_quadruple_data

        assert len(train_texts) == len(train_quadruple_data), \
            "Length samples mismatch between train_texts and train_triple_data"

        if test_sentences:
            test_texts = []
            test_quadruple_data = []
            for sentence in tqdm(test_sentences, desc=f"Make triple data"):
                public_id = sentence.get('id', '')
                content = sentence.get('content', '')
                tags = sentence.get('tags', [])

                if not tags:
                    logger.warning(f'--> Missing label tags in id={public_id} and content={content}')
                    continue

                quadruple = DataParser.get_quadruples(public_id, content, tags)

                if not quadruple:
                    logger.error(f'Quadruple empty in id={public_id} and content={content}')
                    continue

                test_texts.append(content)
                test_quadruple_data.append(quadruple)

            ##Filtering to get each tag as sample
            new_test_texts = []
            new_test_quadruple_data = []
            for i in range(len(test_quadruple_data)):
                for quadruple in test_quadruple_data[i]:
                    new_test_texts.append(test_texts[i])
                    new_test_quadruple_data.append(quadruple)

            test_texts = new_test_texts
            test_quadruple_data = new_test_quadruple_data

        else:
            train_texts, train_quadruple_data, test_texts, test_quadruple_data = DataParser.make_train_test(
                texts=train_texts,
                quadruple_data=train_quadruple_data,
                test_size=0.1
            )


        return train_texts, train_quadruple_data, test_texts, test_quadruple_data

    @staticmethod
    def get_quadruples(public_id, content, tags):
        quadruples = []
        content_words = content.split()

        for tag in tags:
            subject = tag['subject']
            object = tag['object']
            aspect = tag['aspect']
            preference = tag['preference'].upper()

            tmp_subject, tmp_object, tmp_aspect = None, None, None

            if subject:
                tmp_subject = [subject.get('start_offset'), subject.get('end_offset')]
            else:
                tmp_subject = [-1, -1]

            if object:
                tmp_object = [object.get('start_offset'), object.get('end_offset')]
            else:
                tmp_object = [-1, -1]

            if aspect:
                tmp_aspect = [aspect.get('start_offset'), aspect.get('end_offset')]
            else:
                tmp_aspect = [-1, -1]

            quadruples.append((tmp_subject, tmp_object, tmp_aspect, preference, public_id))

        return quadruples

    @staticmethod
    def make_train_test(texts, quadruple_data, test_size: float = 0.1):
        train_size = int((1 - test_size) * len(texts))
        list_index = list(range(len(texts)))
        train_ids = random.sample(list_index, train_size)
        train_ids.sort()
        test_ids = list(set(list_index) - set(train_ids))
        test_ids.sort()

        train_texts = []
        test_texts = []
        train_quadruple_data = []
        test_quadruple_data = []

        for ind in train_ids:
            train_texts.append(texts[ind])
            train_quadruple_data.append(quadruple_data[ind])

        for ind in test_ids:
            test_texts.append(texts[ind])
            test_quadruple_data.append(quadruple_data[ind])

        return train_texts, train_quadruple_data, test_texts, test_quadruple_data

    @staticmethod
    def get_text(lines):
        text_list = []
        for line in lines:
            word_list = line.lower().split()
            text_list.append(word_list)

        return text_list

    @staticmethod
    def fusion(quadruple):
        opinion = []
        for t in quadruple:
            if t[3] not in opinion:
                opinion.append(t[3])

        return opinion

    @staticmethod
    def norm_aspect(aspect):
        if aspect is not None:
            aspect = unl('NFKC', aspect)
            aspect = re.sub('&', ' & ', aspect)
            aspect = aspect.replace('_', ' ')
            aspect = aspect.lower().strip()

        return aspect

    @staticmethod
    def processing_one(texts, quadruple_data, dataset_type='train'):
        standard_data = None
        sample_list = []
        standard_preference_list = []
        standard_preference_text_list = []
        standard_id_list = []
        standard_text_list = []
        standard_subject_list = []
        standard_object_list = []
        standard_aspect_list = []

        text_list = DataParser.get_text(texts)

        header_fmt = 'Processing {:>5s}'
        for i in tqdm(range(len(text_list)), desc=f"{header_fmt.format(dataset_type)}"):
            quadruple = quadruple_data[i]
            text = text_list[i]
            preference = PREFERENCE2ID[quadruple[-2]]


            #TODO: Preference Query
            # query = ''
            # for j in range(3):
            #     elem = quadruple[j]
            #     if elem[0] != -1 and elem[-1] !=-1:
            #         elem_text = ' '.join(text[elem[0]:elem[-1]+1])
            #         query = query + elem_text + ' '
            # query = query[:-1].lower().split()
            for j in range(3):
                elem = quadruple[j]
                if elem[0] == -1 and elem[-1] ==-1:
                    if j == 0:
                        subject_term = '-'
                    elif j == 1:
                        object_term = '-'
                    elif j == 2:
                        aspect_term = '-'
                else:
                    if j == 0:
                        subject_term = ' '.join(text[elem[0]:elem[-1]+1])
                    elif j == 1:
                        object_term = ' '.join(text[elem[0]:elem[-1]+1])
                    elif j == 2:
                        aspect_term = ' '.join(text[elem[0]:elem[-1]+1])
            query = f'which comparative preference given subject {subject_term} , object {object_term} and aspect {aspect_term} ?'.lower().split()

            sample = DualSample(
                texts[i],
                text,
                query,
                [preference]
            )
            sample_list.append(sample)

            standard_preference_list.append(preference)
            standard_preference_text_list.append(quadruple[-2])
            standard_id_list.append(quadruple[-1])
            standard_text_list.append(' '.join(text))
            
            ##Getting standard subject, object and aspect
            for j in range(3):
                elem = quadruple[j]
                if elem[0] != -1 and elem[-1] !=-1:
                    if j == 0:
                        standard_subject_list.append(' '.join(text[elem[0]:elem[-1]+1]))
                    elif j == 1:
                        standard_object_list.append(' '.join(text[elem[0]:elem[-1]+1]))
                    elif j == 2:
                        standard_aspect_list.append(' '.join(text[elem[0]:elem[-1]+1]))
                else:
                    if j == 0:
                        standard_subject_list.append('')
                    elif j == 1:
                        standard_object_list.append('')
                    elif j == 2:
                        standard_aspect_list.append('')





        assert len(standard_preference_list) == len(standard_preference_text_list)

        if dataset_type.lower() in ['dev', 'test']:
            standard_data = {
                'public_ids': standard_id_list,
                'texts': standard_text_list,
                'subjects': standard_subject_list,
                'objects': standard_object_list,
                'aspects': standard_aspect_list,
                'pref_ids': standard_preference_list,
                'pref_text': standard_preference_text_list
            }


        assert len(sample_list) == len(standard_preference_text_list), \
            "Mismatch length of sample_list and length of standard_preference_text_list !"

        return sample_list, standard_data

    @staticmethod
    def convert_examples_to_features(data, tokenizer, dataset_type):
        max_preference_query_length = 0

        #Preference group
        _preference_query = []
        _preference_query_mask = []
        _preference_query_seg = []
        _preference_answer = []

        header_fmt = 'Tokenize {:>5s}'
        for sample in tqdm(data, desc=f"{header_fmt.format(dataset_type)}"):
            temp_pref_query = sample.preference_query
            temp_text = sample.text
            temp_pref_answer = sample.preference_answer

            #Preference group
            pref_query = []
            pref_query_mask = []
            pref_query_seg = []
            pref_query_tokens_ids = []
            text_tokens_ids = []

            for i, word in enumerate(temp_pref_query):
                tokens = tokenizer.encode(word)
                pref_query_tokens_ids.extend(tokens[1: -1])

            pref_query.extend([tokenizer.cls_token_id])
            pref_query.extend(pref_query_tokens_ids)

            for i, word in enumerate(temp_text):
                tokens = tokenizer.encode(word)
                text_tokens_ids. extend(tokens[1:-1])

            pref_query.extend([tokenizer.sep_token_id])
            pref_query_seg = [0] * len(pref_query) + [1] * (len(text_tokens_ids) + 1)
            pref_query.extend(text_tokens_ids)
            pref_query.extend([tokenizer.sep_token_id])
            pref_query_mask = [1] * len(pref_query)

            assert len(pref_query) == len(pref_query_seg) == len(pref_query_mask)
            
            if len(pref_query) > max_preference_query_length:
                max_preference_query_length = len(pref_query)

            _preference_query.append(pref_query)
            _preference_query_seg.append(pref_query_seg)
            _preference_query_mask.append(pref_query_mask)
            _preference_answer.append(temp_pref_answer)

            assert len(_preference_query) == len(_preference_answer)


        for i in range(len(_preference_query)):
            pref_pad_num = max_preference_query_length - len(_preference_query[i])


            _preference_query[i].extend([tokenizer.pad_token_id] * pref_pad_num)
            _preference_query_mask[i].extend([0] * pref_pad_num)
            _preference_query_seg[i].extend([1] * pref_pad_num)

        ##Applying random oversampling or undersampling
        if dataset_type == 'train':
            data_dict = {
                '_preference_query': _preference_query,
                '_preference_query_mask': _preference_query_mask,
                '_preference_query_seg': _preference_query_seg 
            }

            df = pd.DataFrame(data=data_dict, columns=['_preference_query', '_preference_query_mask', '_preference_query_seg'])

            ## Random Over Sampler
            ros = RandomOverSampler(random_state=42)
            ros_df, ros_targets = ros.fit_resample(df, _preference_answer)

            _preference_query = ros_df['_preference_query'].tolist()
            _preference_query_mask = ros_df['_preference_query_mask'].tolist()
            _preference_query_seg = ros_df['_preference_query_seg'].tolist()
            _preference_answer = ros_targets

            ## Random Under Sampler
            # rus = RandomUnderSampler(random_state=42)
            # rus_df, rus_targets = rus.fit_resample(df, _preference_answer)

            # _preference_query = rus_df['_preference_query'].tolist()
            # _preference_query_mask = rus_df['_preference_query_mask'].tolist()
            # _preference_query_seg = rus_df['_preference_query_seg'].tolist()
            # _preference_answer = rus_targets

            # ##SMOTE
            # temp_preference_query = np.array(_preference_query)
            # temp_preference_query_mask = np.array(_preference_query_mask)
            # temp_preference_query_seg = np.array(_preference_query_seg)
            # temp_full_data = np.concatenate((temp_preference_query, temp_preference_query_mask, temp_preference_query_seg), axis=1)
            # sm = SMOTE(random_state=42, k_neighbors=4)
            # sm_data, sm_targets = sm.fit_resample(temp_full_data, _preference_answer)
            # _preference_query = sm_data[:,0:temp_preference_query.shape[1]].tolist()
            # _preference_query_mask = sm_data[:,temp_preference_query.shape[1]:temp_preference_query.shape[1]*2].tolist()
            # _preference_query_seg = sm_data[:, temp_preference_query.shape[1]*2:].tolist()
            # _preference_answer = sm_targets

            # ##ADASYN
            # temp_preference_query = np.array(_preference_query)
            # temp_preference_query_mask = np.array(_preference_query_mask)
            # temp_preference_query_seg = np.array(_preference_query_seg)
            # temp_full_data = np.concatenate((temp_preference_query, temp_preference_query_mask, temp_preference_query_seg), axis=1)
            # ada = ADASYN(random_state=42, n_neighbors=5)
            # ada_data, ada_targets = ada.fit_resample(temp_full_data, _preference_answer)
            # _preference_query = ada_data[:,0:temp_preference_query.shape[1]].tolist()
            # _preference_query_mask = ada_data[:,temp_preference_query.shape[1]:temp_preference_query.shape[1]*2].tolist()
            # _preference_query_seg = ada_data[:, temp_preference_query.shape[1]*2:].tolist()
            # _preference_answer = ada_targets



        result = {
            "_preference_query": _preference_query,
            "_preference_answer": _preference_answer,
            "_preference_query_mask": _preference_query_mask,
            "_preference_query_seg": _preference_query_seg
        }

        return OriginalPreferenceDataset(result)

    @staticmethod
    def run(train_paths, test_path, tokenizer, test_size: float=0.1):
        train_texts, train_quadruple_data, test_texts, test_quadruple_data = DataParser.load_data_from_json(
            train_paths, test_path
        )

        # TODO: Data process
        train_data, _ = DataParser.processing_one(train_texts, train_quadruple_data, dataset_type='train')
        test_data, test_standard_data = DataParser.processing_one(test_texts, test_quadruple_data, dataset_type='test')
        # TODO: Convert examples to features
        train_tokenized = DataParser.convert_examples_to_features(train_data, tokenizer, dataset_type='train')
        test_tokenized = DataParser.convert_examples_to_features(test_data, tokenizer, dataset_type='test')

        return train_tokenized, test_tokenized, test_standard_data
