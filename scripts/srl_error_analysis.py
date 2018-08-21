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


def new_histogram(num_bins, num_stats):
  hist = []
  for i in range(num_bins):
    hist.append([0] * num_stats)
  return hist


def print_histogram(name, p_hist, r_hist, buckets):
  new_p = new_histogram(len(buckets), len(p_hist[0]))
  new_r = new_histogram(len(buckets), len(r_hist[0]))

  for i, p in enumerate(p_hist):
    for bucket_id, b in enumerate(buckets):
      if b[0] <= i and i <= b[1]:
        new_p[bucket_id][0] += p[0]
        new_p[bucket_id][1] += p[1]

  for i, r in enumerate(r_hist):
    for bucket_id, b in enumerate(buckets):
      if b[0] <= i and i <= b[1]:
        new_r[bucket_id][0] += r[0]
        new_r[bucket_id][1] += r[1]

  print ('Precision breakdown by {}:'.format(name))
  for i, b in enumerate(buckets):
    p = new_p[i]
    print ('{}-{}\t{}\t{}'.format(b[0], b[1], p[1], 100.0 * p[0] / p[1]))

  print ('Recall breakdown by {}:'.format(name))
  for i, b in enumerate(buckets):
    r = new_r[i]
    print ('{}-{}\t{}\t{}'.format(b[0], b[1], r[1], 100.0 * r[0] / r[1]))

  print ('F1 breakdown by {}:'.format(name))
  for i, b in enumerate(buckets):
    p = 100.0 * new_p[i][0] / new_p[i][1]
    r = 100.0 * new_r[i][0] / new_r[i][1]
    f1 = 2 * p * r / (p + r)
    print ('{}-{}\t{}'.format(b[0], b[1], f1))


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

  prc_by_arg_len = new_histogram(200, 2)
  rec_by_arg_len = new_histogram(200, 2)
  prc_by_dist_to_prop = new_histogram(200, 2)
  rec_by_dist_to_prop = new_histogram(200, 2)

  def _get_surface_dist(pred_id, start, end):
    return min(abs(pred_id - start), abs(pred_id - end))

  for sent_id, tokens in enumerate(all_tokens):
    predicted_pas = predicted_arguments[sent_id]
    gold_pas = gold_arguments[sent_id]
    predicted_preds = set()
    for pred, args in predicted_pas.iteritems():
      none_v_args = [a for a in args if a[2] not in ['V', 'C-V']]
      pas_f1.num_predicted += len(none_v_args)
      # Add predicted args to analysis.
      for start, end, _ in none_v_args:
        alen = end - start + 1
        dist = _get_surface_dist(pred, start, end)
        prc_by_arg_len[alen][1] += 1
        prc_by_dist_to_prop[dist][1] += 1

      # Count matched arguments.
      if pred in gold_pas:
        matched_args = [a for a in gold_pas[pred] if a in none_v_args]
        pas_f1.num_matched += len(matched_args)
        # Add matched args to analysis.
        for start, end, _ in matched_args:
          alen = end - start + 1
          dist = _get_surface_dist(pred, start, end)
          prc_by_arg_len[alen][0] += 1
          rec_by_arg_len[alen][0] += 1
          prc_by_dist_to_prop[dist][0] += 1
          rec_by_dist_to_prop[dist][0] += 1

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
        # Add gold args to analysis.
        for start, end, _ in none_v_args:
          alen = end - start + 1
          dist = _get_surface_dist(pred, start, end)
          rec_by_arg_len[alen][1] += 1
          rec_by_dist_to_prop[dist][1] += 1

    # Print human-readable.
    print (' '.join(tokens))
    print ('===Predicted===')
    conll_io.PrintHumanReadable(tokens, None, predicted_pas)
    print ('======Gold=====')
    conll_io.PrintHumanReadable(tokens, None, gold_pas)
    print ('\n')

  print ('Predicate ID F1\n' + predicate_f1.print_info())
  print ('Non-official SRL F1\n' + pas_f1.print_info())

  print_histogram('Distance to predicate',
                  prc_by_dist_to_prop, rec_by_dist_to_prop,
                  [(0,1), (2,3),(4,7),(8,200)])
  print_histogram('Argument length',
                  prc_by_arg_len, rec_by_arg_len,
                  [(0,1), (2,4), (5,10), (10,200)])

if __name__ == '__main__':
  app.run(main)
