"""TODO(luheng): DO NOT SUBMIT without one-line documentation for srl_error_analysis.

TODO(luheng): DO NOT SUBMIT without a detailed description of srl_error_analysis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import conll_io_utils as conll_io
import json

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'predicted_conll_input', './ontonotes_data/v5.dev.props.gold.txt',
    'String path to predicted CoNLL file.')

flags.DEFINE_string(
    'gold_conll_input', './ontonotes_data/v5.dev.props.gold.txt',
    'String path to gold CoNLL file.')

flags.DEFINE_string(
    'json_input', './ontonotes_data/dev.english.mtl.jsonlines',
    'String path to the json input data.')


class F1(object):
  def __init__(self):
    self.num_predicted = 0
    self.num_gold = 0
    self.num_matched = 0

  def recall(self):
    if self.num_gold == 0:
      return 0.0
    return 1.0 * self.num_matched / self.num_gold

  def precision(self):
    if self.num_predicted == 0:
      return 0.0
    return 1.0 * self.num_matched / self.num_predicted

  def f1(self):
    if self.num_matched == 0:
      return 0.0
    p = self.precision()
    r = self.recall()
    return 2 * p * r / (p + r)

  def print_info(self):
    prt = 'Counts (g/p/m): {}/{}/{}\n'.format(
        self.num_gold, self.num_predicted, self.num_matched)
    prt += 'Precision: ' + str(self.precision()) + '\n'
    prt += 'Recall: ' + str(self.recall()) + '\n'
    prt += 'F1: ' + str(self.f1()) + '\n'
    return prt


def ReadTokens(json_input_path):
  all_tokens = []  # List of sentences.
  with open(json_input_path, 'r') as json_input:
    for line in json_input.readlines():
      doc_info = json.loads(line.strip())
      all_tokens.extend(doc_info['sentences'])
    json_input.close()
  print('Read {} sentences.'.format(len(all_tokens)))
  return all_tokens


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Read tokens.
  all_tokens = ReadTokens(FLAGS.json_input)

  # Read predictions.
  gold_predicates, gold_arguments = conll_io.ReadFromCoNLL(FLAGS.gold_conll_input)
  predicted_predicates, predicted_arguments = conll_io.ReadFromCoNLL(
      FLAGS.predicted_conll_input)

  assert len(gold_predicates) == len(gold_arguments)
  assert len(gold_predicates) == len(predicted_arguments)
  assert len(gold_predicates) == len(predicted_predicates)

  predicate_f1 = F1()
  pas_f1 = F1()
  num_sentences = 0
  for sent_id, tokens in enumerate(all_tokens):
    predicted_pas = predicted_arguments[sent_id]
    gold_pas = gold_arguments[sent_id]
    predicted_preds = set()
    for pred, args in predicted_pas.iteritems():
      print (args)
      none_v_args = [a for a in args if a[2] not in ['V', 'C-V']]
      pas_f1.num_predicted += len(none_v_args)
      # Count matched arguments.
      if pred in gold_pas:
        matched_args = [a for a in gold_pas[pred] if a in none_v_args]
        pas_f1.num_matched += len(matched_args)
      if none_v_args:
        predicted_preds.add(pred)
    predicate_f1.num_predicted += len(predicted_preds)
    # Count gold arguments.
    for pred, args in gold_pas.iteritems():
      none_v_args = [a for a in args if a[2] not in ['V', 'C-V']]
      pas_f1.num_gold += len(none_v_args)
      if none_v_args:
        predicate_f1.num_gold += 1
        if pred in predicted_preds:
          predicate_f1.num_matched += 1
    # Print human-readable.
    print (' '.join(tokens))
    print ('===Predicted===')
    conll_io.PrintHumanReadable(tokens, None, predicted_pas)
    print ('======Gold=====')
    conll_io.PrintHumanReadable(tokens, None, gold_pas)
    print ('\n')

  print ('Predicate ID F1\n' + predicate_f1.print_info())
  print ('Non-official SRL F1\n' + pas_f1.print_info())


if __name__ == '__main__':
  app.run(main)
