# This file defines the NerTrainer class, which is a subclass of Seq2SeqTrainer.
# It is used to train| evaluate a Named Entity Recognition (NER) model.

import itertools
import numpy as np
from transformers import Seq2SeqTrainer, BartForConditionalGeneration
from transformers.utils import logging
from torch import nn
from src import func_util

loger = logging.get_logger(__name__)

class OadaTrainer(Seq2SeqTrainer):
    def __init__(self, label2id, id2label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label2id = label2id
        self.id2label = id2label
        self.extra_data = None
        self.compute_metrics = self.compute_custom_metrics

    def extract_entities(self, decoded_preds: list[str], input_sents: list[str]):
        """
        Extract entities from the decoded prediction.
        :param decoded_preds: list[str], The decoded predictions.
        :param input_sents: list[str], The input sentences.
        :return:
        """
        entities = []  # store the entities in a batch
        for sent, decoded_pred in zip(input_sents, decoded_preds):
            results = decoded_pred.split('|')
            instance_entities = []  # store the entities in a sentence
            for res in results:  # split the result by '|' to get the entities.
                if res and res !=' ':
                    answers = res.split(',')  # split the answers by ','. The format is 'entity_mention,entity_label'.
                    if len(answers) < 2:
                        continue
                    mention, label = answers[:2]  # get the first two items, which are entity mention and entity label.
                    mention, label = mention.strip(), label.strip()
                    founded_spans = func_util.find_span(sent, mention)  # find the span of the entity mention in the sentence.
                    for start, end, span in set(founded_spans):
                        if label not in self.label2id.keys():
                            # skip the label that is not in the label2id mapping.
                            continue
                        # covert label to id
                        out_label_id = self.label2id[label]
                        instance_entities.append([str(start), str(end), span, str(out_label_id)])
                        # instance_entities.append([start, end, out_label_id])  # no mention span
            entities.append(instance_entities)
        return entities

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Get inputs from needed column and Compute the loss of the model.
        :param model: The model to be trained.
        :param inputs: The input data.
        :param return_outputs: Whether to return the outputs.
        :param num_items_in_batch: The number of items in the batch.
        :return:
        """
        # 1. get needed columns
        needed_inputs = {k: v for k, v in inputs.items() if k in self._signature_columns}

        # 2. forward to get loss and logits
        outputs = model(**needed_inputs)
        xe_loss = outputs.loss  # cross-entropy loss for a batch
        logits = outputs.logits  # logits for a batch. The shape is (batch_size, seq_len, vocab_size)

        # 3. get the OADA-XE
        # Actually, we need to get all rearrangement O in training.
        # But each instance has different size of O, meanwhile the batch size is fixed.
        # So we cannot get each ordering to calcualte the OADA-XE loss.
        # Further, we cannot get best ordering to minimize the XE loss when using batch training.
        # Here, we just get the minimum XE loss as OADA-XE loss approximately.
        log_softmax = nn.LogSoftmax(dim=-1)
        predictions = log_softmax(logits)  # predictions shape: (batch_size, seq_len, vocab_size)
        batch_size, seq_len, vocab_size = predictions.shape

        labels = inputs ['labels']  # shape: (batch_size, seq_len)
        nll_loss = nn.NLLLoss(reduction='none')
        oada_losses = nll_loss(
            predictions.view(-1, vocab_size),
            labels.view(-1),
        )  # shape: (batch_size * seq_len, )
        oada_losses = oada_losses.reshape(batch_size, seq_len)
        oada_losses = oada_losses.sum(dim=-1)  # shape: (batch_size, )
        oada_loss = oada_losses.min()  # get the minimum OADA loss in a batch

        if len(self.state.log_history) == 0:
            training_step = 0
        else:
            training_step = self.state.log_history[-1]['step']  # current training step
        max_steps = self.state.max_steps

        # 4. get the total loss
        oada_xe_weight = training_step / max_steps
        loss = (1 - oada_xe_weight) * xe_loss + oada_xe_weight * oada_loss

        return (loss, outputs) if return_outputs else loss

    def setup_extra_for_metric(self, extra_data):
        """
        Setup extra data for metric computation.
        :param extra_data: The extra data to be used in metric computation.
        :return:
        """
        self.extra_data = extra_data

    def compute_custom_metrics(self, eval_pred):
        """
        Compute the confusion matrix, span-level metrics such as precision, recall and F1-micro.
        :param eval_pred: EvalPrediction object, including
            1) predictions, is the generated token ids when set predict_with_generate=True in TrainingArguments.
            2) label_ids, is the label token ids.
            3) inputs (if set include_for_metrics in TrainingArguments).
            to be decoded.
        :return:
        """
        # get needed data
        preds, label_ids = eval_pred

        if self.is_in_train:  # if in training, we need to get self.extra_data['validation'].
            spans_labels = self.extra_data['span_labels']['validation']
            input_sents = self.extra_data['input_sents']['validation']
        else:  # if in evaluation, we need to get self.extra_data['test'].
            spans_labels = self.extra_data['span_labels']['test']
            input_sents = self.extra_data['input_sents']['test']
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, self.processing_class.pad_token_id)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=True)
        # loger.info(f'decoded_preds:\n {decoded_preds}')

        # extract entities from the decoded predictions and get their positions according to the input sentences.
        # pred_spans shapes like [[start, end, span, label_id], [...], ...]
        pred_spans = self.extract_entities(decoded_preds, input_sents)
        # assert len(pred_spans) == len(spans_labels), 'The number of pred_spans and spans_labels should be the same.'
        # loger.info(f'pred_spans:\n {pred_spans}')
        # loger.info(f'gold_spans:\n {spans_labels}')

        # filter empty pred in the pred_spans and filter 'O' label
        # then flatten the pred_spans
        pred_spans = [
            span
            for instance_spans in pred_spans
            if instance_spans  # filter empty pred in the pred_spans
            for span in instance_spans
            if int(span[-1]) != self.label2id['O']  # filter 'O' label
        ]
        gold_spans = list(itertools.chain(*spans_labels)) # flatten gold_spans
        result = func_util.compute_span_f1(gold_spans, pred_spans)
        return result
