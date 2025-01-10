# We provide the Processor class to process the data.
# MyDataCollatorForSeq2Seq is used to collate the data for the Seq2Seq model in training and evaluation stages.

import os
import copy
import random
import math
import jsonlines
import multiprocess

from src import func_util
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from transformers.utils import logging

logger = logging.get_logger('Processor')

class Processor:
    """
    The Processor class is used to process the data.
    """
    def __init__(self, config, tokenizer):
        """
        Initialize the Processor class.
        :param config: the config file for running python script. config.dataset is the dataset config.
        :param tokenizer: the tokenizer for the model.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.data_cfg = config.dataset
        self.labels = self.data_cfg['labels']
        self.raw_label2id = self.data_cfg['raw_label2id']  # the raw label2id mapping from the raw dataset.
        self.raw_bio = self.data_cfg['raw_bio']  # a flag to indicate whether the labels are in BIO format in the raw dataset.
        self.label2id = dict()
        self.id2label = dict()
        self.covert_tag2id = dict() if self.raw_bio else None  # covert the original BIO label (tag) id to the new label id. e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
        if self.config.natural_label:  # use natural-language-form labels
            self.init_natural_labels()
        else:  # use simple-form labels
            self.init_simp_labels()
        self.natural_flag = 'natural' if self.config.natural_label else 'bio'  # use natural-form or bio-form

    def init_simp_labels(self):
        """
        init label2id, id2label, covert_tag2id from label_cfgs using simple-form.
        All label_cfgs are simplified format. e.g., person -> PER, location -> LOC, organization -> ORG.
        :return:
        """
        idx = 0
        for k, v in self.raw_label2id.items():
            if k.startswith('B-') or k.startswith('I-'):
                label = k.replace('B-', '').replace('I-', '')
            else:
                label = k
            if label not in self.label2id.keys():
                self.label2id[label] = idx
                self.id2label[idx] = label
                idx += 1

            # if the labels are not in BIO format in the raw datasets, there is no need to covert the
            # original BIO label (tag) id to the new label id.
            if self.raw_bio and v not in self.covert_tag2id.keys():
                # e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
                self.covert_tag2id[v] = self.label2id[label]

    def init_natural_labels(self):
        """
        init label2id, id2label, covert_tag2id from label_cfgs using natural language form.
        ALL label_cfgs are natural format. e.g., person, location, organization.
        :return:
        """
        self.label2id['O'] = 0
        self.id2label[0] = 'O'
        for index, (k, v) in enumerate(self.labels.items()):
            label = v['natural']
            id = index + 1
            self.label2id[label] = id
            self.id2label[id] = label

        # if the labels are not in BIO format in the raw datasets, there is no need to covert the
        # original BIO label (tag) id to the new label id.
        if not self.raw_bio:
            return

        # if the labels are in BIO format in the raw datasets, covert the original BIO label (tag) id to the new label id.
        self.covert_tag2id[0] = 0  # 'O' -> 'O'
        for k, v in self.raw_label2id.items():
            if k.startswith('B-') or k.startswith('I-'):
                label = k.replace('B-', '').replace('I-', '')
            else:
                label = k
            if label not in self.labels.keys():  # skip 'O'
                continue
            natural_label = self.labels[label]['natural']
            self.covert_tag2id[v] = self.label2id[natural_label]

    def _get_span_and_labels(self, tokens, tags):
        """
        Get the span and span label of the sentence, given the tokens and token tags.
        :param tokens: tokens of the sentence
        :param tags: tags for each token
        :return:
        """
        spans_labels = []  # store spans and labels for each instance
        idx = 0
        span = []  # store tokens in a span
        pre_tag = 0  # the previous tag
        start, end = 0, 0  # the start/end index for a span

        while idx < len(tokens):
            tag = tags[idx]
            if tag != 0:
                if pre_tag != 0 and self.covert_tag2id[tag] == self.covert_tag2id[pre_tag]:  # the token is in the same span
                    # append the token into the same span
                    span.append(tokens[idx])
                    end = idx + 1  # exclusive
                else:  # the previous is a 'O' token or previous token is not in the same span
                    # store the previous span
                    if len(span) > 0:
                        span_tag = tags[start]  # the label of the span, we use the label of the first token in the span
                        # an element in instance_spans_labels is like (start, end (excluded), gold_mention_span, gold_label_id)
                        spans_labels.append((str(start), str(end), ' '.join(span), str(self.covert_tag2id[span_tag])))
                    # init a new span
                    span.clear()
                    span.append(tokens[idx])
                    start = idx
                    end = idx + 1  # exclusive
            pre_tag = tag
            idx += 1
        # store the last span
        if len(span) > 0:
            spans_labels.append((str(start), str(end), ' '.join(span), str(self.covert_tag2id[tags[start]])))
        return spans_labels

    def data_format_span(self, instances):
        """
        Get the span from gold annotated spans.
        :param instances: Dict[str, List], A batch of instances.
        :return:
        """

        # init the result
        res_sents = []  # store the sentence of the instances
        res_spans_labels = []  # store the gold spans and labels of the instances
        res_tgt_sequence = []  # store the target sequence of the instances

        # main process
        tokens_filed, ner_tags_field = self.data_cfg['tokens_field'], self.data_cfg['ner_tags_field']
        all_raw_tokens, all_raw_tags = instances[tokens_filed], instances[ner_tags_field]
        # 1. Some preparations
        # 1.2. covert tokens to sentence
        sents = [' '.join(raw_tokens) for raw_tokens in all_raw_tokens]

        # 1.3. get batch for different settings
        if not self.data_cfg['nested']:  # flat ner
            pbar = zip(sents, all_raw_tokens, all_raw_tags)
        else:  # nested
            start_position, end_position = instances['starts'], instances['ends']
            pbar = zip(sents, all_raw_tokens, start_position, end_position, all_raw_tags)

        # 2. process each instance
        for instance in pbar:
            # 2.1 (optional) get the tag directly from the raw dataset
            spans_labels = []  # store gold spans and their labels for this instance
            tgt_sequence = ''  # store the target sequence for this instance
            if not self.data_cfg['nested']:  # flat NER
                sent, raw_tokens, raw_tags = instance

                # 2.1 get gold spans and their labels  for this instance
                spans_labels = self._get_span_and_labels(raw_tokens, raw_tags)

                # element in gold_spans is in the shape of (str(start), str(end) (excluded), span)
                # element in gold_spans_tags is tag id
                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, gold_label_id)...]
                for start, end, mention, label_id in spans_labels:
                    # an element in instance_spans_labels is like (start, end (excluded), gold_mention_span, gold_label_id)
                    tgt_sequence += '{}, {}| '.format(mention, self.id2label[int(label_id)])

            else:  # nested NER
                sent, raw_tokens, starts, ends, raw_tags = instance

                for start, end, label_id in zip(starts, ends, raw_tags):
                    # end position is excluded
                    mention = ' '.join(raw_tokens[start: end])
                    spans_labels.append((str(start), str(end), mention, str(label_id)))
                    tgt_sequence += '{}, {}| '.format(mention, self.id2label[int(label_id)])

            res_sents.append(sent)
            res_spans_labels.append(spans_labels)
            res_tgt_sequence.append(tgt_sequence)

        return {
            'sentence': res_sents,
            'spans_labels': res_spans_labels,  # store the gold spans and labels of the instances, shape like (start, end (excluded), gold_mention_span, gold_label_id)
            'tgt_sequence': res_tgt_sequence,  # store the target sequence of the instances for seq2seq model
        }

    def data_augmentation_by_oada(self, instances):
        """
        Get the span from gold annotated spans using order-agnostic data augmentation.
        :param instances: Dict[str, List], A batch of instances.
        :return:
        """

        # init the result
        res_sents = []  # store the sentences of the instances
        res_spans_labels = []  # store the gold spans and labels of the instances
        res_tgt_sequence = []  # store the target sequence of the instances

        # 1. get label subset if we use 'label_subset' augmentation
        assert 'rearrangement_num' in self.config, \
            '"rearrangement_num" should be provided in the kwargs for order-agnostic data augmentation.' \
            'Please check the specify "rearrangement_num" parameters in config file or command line.'

        all_labels = list(self.label2id.keys())
        if 'O' in all_labels:
            all_labels.remove('O')

        label_arrangements = func_util.get_entity_arrangement(
            all_labels,
            self.config['rearrangement_num']
        )

        # 2. process each instance
        # an element of instances is a dict, containing the id, tokens, spans_labels and tgt_sequence
        for raw_sentence, raw_spans_labels, raw_tgt_sequence in zip(instances['sentence'], instances['spans_labels'], instances['tgt_sequence']):

            # group span labels according to label
            grouped_spans_labels = {label: [] for label in all_labels}
            for start, end, mention, label_id in raw_spans_labels:
                label = self.id2label[int(label_id)]
                grouped_spans_labels[label].append((start, end, mention, label_id))

            # order-agnostic data augmentation
            for label_arrangement in label_arrangements:

                tgt_sequence = ''  # store the new target sequence for this instance
                spans_labels = []  # store the spans and labels using label subset partition
                tokens = copy.deepcopy(raw_sentence)  # store the tokens using label subset partition

                for label in label_arrangement:
                    if len(grouped_spans_labels[label]) == 0:
                        continue
                    for start, end, mention, label_id in grouped_spans_labels[label]:
                        # an element in instance_spans_labels is like (start, end (excluded), gold_mention_span, gold_label_id)
                        tgt_sequence += '{}, {}| '.format(mention, label)
                    spans_labels += grouped_spans_labels[label]

                # if not using negative sampling, we should skip this instance
                # when using negative sampling and there is no golden entity spans in the instance (tgt_sequence is '')
                if not self.config.negative and len(spans_labels) == 0:
                    continue
                order_instruction = ','.join(label_arrangement)
                tokens += ' [' + order_instruction + ']' # concat order_instruction to the tokens
                res_sents.append(tokens)
                res_spans_labels.append(spans_labels)
                res_tgt_sequence.append(tgt_sequence)

        return {
            'sentence': res_sents,
            'spans_labels': res_spans_labels,
            # store the gold spans and labels of the instances, shape like (start, end (excluded), gold_mention_span, gold_label_id)
            'tgt_sequence': res_tgt_sequence,  # store the target sequence of the instances for seq2seq model
        }

    def statistics(self, dataset, include_none=False):
        """
        Get the statistics of the dataset.
        :param dataset: the dataset to be analyzed.
        :param include_none: whether to include the instances without any golden entity spans. True means to include.
        :return: the statistics of the dataset.
        """
        # get the statistics of the dataset
        # check the cached
        # 1.1 get the entity number of each label

        label_nums = {k: 0 for k in self.label2id.keys() if k != 'O'}  # store the number of entities for each label
        label_indices = {k: [] for k in self.label2id.keys() if k != 'O' }  # store the index of instances for each label

        if include_none:
            label_nums['none'], label_indices['none'] = 0, []  # store the number and index of instances without any golden entity spans

        for instance in dataset:
            if include_none and len(instance['spans_labels']) == 0:
                label_nums['none'] += 1
                label_indices['none'].append(instance['id'])
                continue

            for spans_label in instance['spans_labels']:
                # shape like (start, end, gold_mention_span, gold_label_id)
                label_id = int(spans_label[-1])
                label = self.id2label[label_id]
                label_nums[label] += 1
                label_indices[label].append(instance['id'])

        # remove dunplicate indices
        for k, v in label_indices.items():
            label_indices[k] = list(set(v))

        sum_labels = sum(label_nums.values())
        label_dist = {k: v / sum_labels for k, v in label_nums.items()}

        return {
            'label_nums': label_nums,
            'label_dist': label_dist,
            'label_indices': label_indices
        }

    def support_set_sampling(self, dataset, k_shot=1, sample_split='train'):
        """
        Sample k-shot support set (only data index) from the dataset split.
        The sampled support set contains at least K examples for each of the labels.
        Refer to in the Support Set Sampling Algorithm in the Appendix B (P12) of the paper https://arxiv.org/abs/2203.08985
        or in the Algorithm 1 in the A.2 (P14) of the paper https://arxiv.org/abs/2303.08559

        :param dataset: The dataset to be sampled.
        :param k_shot: The shot number of the support set.
        :param sample_split: The dataset split you want to sample from.
        :return: the support set containing k-shot index of examples for each of the labels.
        """
        def _update_counter(support_set, raw_counter):
            """
            Update the number for each label in the support set.
            :param support_set: the support_set
            :param raw_counter: the counter to record the number of entities for each label in the support set
            :return:
            """
            counter = {label: 0 for label in raw_counter.keys()}
            for idx in support_set:
                for spans_label in dataset['spans_labels'][idx]:
                    # spans_label shapes like (start, end, gold_mention_span, gold_label)
                    label_id = int(spans_label[-1])
                    label = self.id2label[label_id]
                    counter[label] += 1
            return counter

        # 1. init
        if sample_split not in dataset.keys():
            dataset = dataset['train']
        else:
            dataset = dataset[sample_split]

        label_nums = self.statistics(dataset)['label_nums']  # count the number of entities for each label
        label_nums = dict(sorted(label_nums.items(), key=lambda x: x[1], reverse=False))  # sort the labels by the number of entities by ascending order

        # add new_tags column
        # original tags is BIO schema, we convert it to the new tags schema where the 'O' tag is 0, 'B-DATE' and 'I-DATE' are the same tag, etc.
        ner_tags_field = self.data_cfg['ner_tags_field']
        if not self.data_cfg['nested']:  # flat NER
            dataset = dataset.map(lambda example: {"new_tags": [self.covert_tag2id[tag] for tag in example[ner_tags_field]]})
        else:  # nested NER
            dataset = dataset.map(lambda example: {"new_tags": [tag for tag in example[ner_tags_field]]})

        support_set = set()  # the support set
        counter = {label: 0 for label in label_nums.keys()}  # counter to record the number of entities for each label in the support set

        # init the candidate instances indices for each label
        candidate_idx = dict()

        for label in label_nums.keys():
            # for 'O' label, we choose those instance containing spans parsed by parsers without any golden entity spans,
            # i.e., len(dataset[idx]['spans']) > 0 and len(dataset[idx]['spans_labels']) <= 0
            # tmp_ins = dataset.filter(lambda x: len(x['spans']) > 0 >= len(x['spans_labels']))['id']

            # filter out the instances without any golden entity spans
            label_id = self.label2id[label]
            tmp_ins = dataset.filter(lambda x: label_id in x['new_tags'] and len(x['spans_labels']) > 0)['id']
            candidate_idx.update({label: tmp_ins})

        # 2. sample
        logger.info(f"Sampling {k_shot}-shot support set from {sample_split} split...")
        for label in label_nums.keys():
            while counter[label] < k_shot:
                if len(candidate_idx[label]) == 0:
                    # if the number of entities for any label in the support set is less than k_shot
                    # we should break the loop to sample another label
                    break
                idx = random.choice(candidate_idx[label])
                support_set.add(idx)
                candidate_idx[label].remove(idx)
                counter = _update_counter(support_set, counter)
                logger.info(f'support set statistic: {counter}')

        # 3. remove redundant instance
        raw_support_set = copy.deepcopy(support_set)
        for idx in tqdm(raw_support_set, desc='removing redundant instance'):
            tmp_support_set = copy.deepcopy(support_set)  # cache before removing instance idx
            support_set.remove(idx)
            counter = _update_counter(support_set, counter)
            # if we remove instance idx, leading to the number of entities for any label in the support set is less than k_shot
            # we should add instance idx back to the support set
            if len(list(filter(lambda e: e[1] < k_shot, counter.items()))) != 0:
                support_set = tmp_support_set

        counter = _update_counter(support_set, counter)
        return support_set, counter

    def subset_sampling(self, dataset: Dataset, size=200, sampling_strategy='random', seed=None):
        """
        Get the subset of the dataset according to sampling sampling_strategy.
        :param dataset: the dataset to be sampled to get subset.
        :param size: the size of the test subset.
        :param sampling_strategy: the sampling strategy.
            1) 'random' for random sampling. Select instances randomly. Each instance has the same probability of being selected.
            2) 'lab_uniform' for uniform sampling at label-level. Choice probability is uniform for each label.
            3) 'proportion' for proportion sampling. Choice probability is proportional to the number of entities for each label.
            4) 'shot_sample' for sampling test set like k-shot sampling. Each label has at least k instances.
        :param seed: the seed for random sampling. If None, a random seed will be used.
        :return:
        """
        assert sampling_strategy in ('random', 'lab_uniform', 'proportion', 'shot_sample')

        if sampling_strategy == 'random':
            if not seed or isinstance(seed, str):
                seed = random.randint(0, 512)
            logger.info(f"Random sampling with seed {seed}...")
            # https://huggingface.co/docs/datasets/process#shuffle
            # use Dataset.flatten_indices() to rewrite the entire dataset on your disk again to remove the indices mapping
            dataset_subset = dataset.shuffle(seed=seed).flatten_indices().select(range(size))

        elif sampling_strategy == 'proportion':
            statistics_res = self.statistics(dataset)
            label_dist,  label_indices= statistics_res['label_dist'], statistics_res['label_indices']
            choice_indices = []
            for label, proportion in label_dist.items():
                choice_num = math.ceil(proportion * size)
                choice_indices += random.sample(label_indices[label], choice_num)

            choice_indices = list(set(choice_indices))
            dataset_subset = dataset.select(choice_indices)

        elif sampling_strategy == 'lab_uniform':
            label_num = len(self.label2id.keys()) - 1  # exclude 'O' label
            statistics_res = self.statistics(dataset)
            label_indices = statistics_res['label_indices']
            choice_indices = []
            for label, indices in label_indices.items():
                choice_num = math.ceil(size / label_num)
                choice_indices += random.sample(indices, choice_num)
            choice_indices = list(set(choice_indices))
            dataset_subset = dataset.select(choice_indices)

        elif sampling_strategy == 'shot_sample':
            support_set, counter = self.support_set_sampling(dataset, k_shot=20, sample_split='train')
            dataset_subset = dataset.select(list(support_set))

        return dataset_subset

    def tokenize_pad_data(self, instances):
        """
        Tokenize data and padding data.
        :param instances: Instances to be processed. We tokenize the 'sentence' and 'tgt_sequence' columns.

        :return:
        """
        inputs = self.tokenizer(instances['sentence'], truncation=True, padding=True)
        labels = self.tokenizer(instances['tgt_sequence'], truncation=True, padding=True)
        inputs['labels'] = labels['input_ids']  # get tgt_sequence's input_ids as labels
        return inputs

    def preprocess(self, seed=22):
        """
        Pre-process the dataset.
        :param seed: the seed for random sampling. If None, a random seed will be used.
        :return: the preprocessed dataset.

        """
        # 0. init config
        preprocessed_dir = os.path.join(self.data_cfg['preprocessed_dir'], f'span_{self.natural_flag}')
        ss_cache_dir = os.path.join(self.data_cfg['ss_cache_dir'], f'span_{self.natural_flag}')  # the directory to cache the support set

        # set 'spawn' start method in the main process to parallelize computation across several GPUs when using multi-processes in the map function
        # refer to https://huggingface.co/docs/datasets/process#map
        multiprocess.set_start_method('spawn')

        # 1. check and load the cached formatted dataset
        try:
            logger.info('Cache found. Load the preprocessed dataset from the cache...')
            dataset = load_from_disk(preprocessed_dir)
        except FileNotFoundError:
            logger.info('No cache found, start to preprocess the dataset...')

            # 2. preprocess data
            # load dataset
            dataset = load_dataset(self.data_cfg['file_path'], num_proc=self.data_cfg['num_proc'], trust_remote_code=True)

            # 2.1. For flat dataset, filter out those instances with different length of tokens and tags
            if not self.data_cfg['nested']:
                # for those flat datasets, we need to filter out those instances with different length of tokens and tags
                tokens_filed, ner_tags_field = self.data_cfg['tokens_field'], self.data_cfg['ner_tags_field']
                dataset = dataset.filter(
                    lambda x: len(x[tokens_filed]) == len(x[ner_tags_field]))

            # 2.2. add index column
            dataset = dataset.map(
                lambda example, index: {"id": index},
                with_indices=True
            )

            # 2.3. format the spans and labels
            dataset = dataset.map(
                self.data_format_span,
                batched=True,
                batch_size=self.data_cfg['data_batch_size'],
                num_proc=self.data_cfg['num_proc'],
                remove_columns=dataset['train'].column_names  # remove the original columns before adding new columns
            )

            # 3. shuffle, split
            if 'validation' in dataset.keys() and 'test' not in dataset.keys():
                # split the origianl 'valid' dataset into 'valid' and 'test' datasets
                valid_test_split = dataset['validation'].train_test_split(test_size=0.5, seed=seed)
                dataset['validation'] = valid_test_split['train']
                dataset['test'] = valid_test_split['test']

            # 4. check the cached the formated data
            if not os.path.exists(preprocessed_dir):
                os.makedirs(preprocessed_dir)
            dataset.save_to_disk(preprocessed_dir)

        # 5. sample the support set
        if self.config.k_shot > 0:
            # If k_shot > 0, the support set will be returned as train split.
            # Else, the preprocessed dataset original train split will be returned.

            if not os.path.exists(ss_cache_dir):
                os.makedirs(ss_cache_dir)

            cache_ss_file_name = 'support_set_{}_shot.jsonl'.format(self.config.k_shot)
            cache_counter_file_name = 'counter_{}_shot.txt'.format(self.config.k_shot)
            support_set_file = os.path.join(ss_cache_dir, cache_ss_file_name)
            counter_file = os.path.join(ss_cache_dir, cache_counter_file_name)

            # check and load the cache
            if not os.path.exists(support_set_file):
                # sample support set from scratch
                # get support set (only data index) and counter
                support_set, counter = self.support_set_sampling(dataset, self.config.k_shot)

                # cache the support set
                with jsonlines.open(support_set_file, mode='w') as writer:
                    for idx in support_set:
                        sentence = dataset['train']['sentence'][idx]
                        spans_labels = dataset['train']['spans_labels'][idx]
                        tgt_sequence = dataset['train']['tgt_sequence'][idx]
                        writer.write({
                            'id': idx,
                            'sentence': sentence,
                            'spans_labels': spans_labels,
                            'tgt_sequence': tgt_sequence,
                        })

                # cache the counter
                with open(counter_file, 'w') as writer:
                    for k, v in counter.items():
                        writer.write(f'{k}: {v}\n')

                # get the support set data
                support_set_data = dataset['train'].select(list(support_set))
            else:
                # load the cached support set
                # load_dataset return a DatasetDict, we need to get the 'train' split
                support_set_data = load_dataset('json', data_files=support_set_file)['train']

            dataset['train'] = support_set_data.remove_columns(['id'])  # replace the original train split with the support set

        # 6. augmentation (optional)
        if self.config.augmentation != 'baseline':
            aug_processed_dir = os.path.join(
                self.data_cfg['preprocessed_dir'],
                f'span_{self.natural_flag}_{self.config.augmentation}'
            )
            try:
                dataset = load_from_disk(aug_processed_dir)
            except FileNotFoundError:
                dataset = dataset.map(
                    self.data_augmentation_by_oada,
                    batched=True,
                    batch_size=self.data_cfg['data_batch_size'],
                    num_proc=self.data_cfg['num_proc'],
                    remove_columns=dataset['train'].column_names  # remove the original columns before adding new columns
                )
                dataset.save_to_disk(aug_processed_dir)

        # 7. tokenize and padding for training and evaluation
        original_columns = dataset['train'].column_names  # store the original columns
        dataset = dataset.map(self.tokenize_pad_data, batched=True)

        # 8. get spans_labels, input_sents for metric computation
        spans_labels = {
            'train': dataset['train']['spans_labels'],
            'validation': dataset['validation']['spans_labels'],
            'test': dataset['test']['spans_labels']
        }
        input_sents = {
            'train': dataset['train']['sentence'],
            'validation': dataset['validation']['sentence'],
            'test': dataset['test']['sentence']
        }
        extra_data = {
            'span_labels': spans_labels,
            'input_sents': input_sents
        }

        # 9. remove original columns.
        # only keep input_ids, attention_mask, labels columns et.al. used to model forward.
        dataset = dataset.remove_columns(original_columns)

        return dataset, extra_data
