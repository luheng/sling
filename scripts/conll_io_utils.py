"""Convert Sling records to CoNLL format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from collections import Iterable
import os
import sling
import subprocess


def PrintHumanReadable(tokens, predicates, arguments, span_is_exclusive=False):
  """ Print SRL predictions into human-readable format.
  """
  # Print predicate-level information.
  sorted_pred_ids = sorted(arguments.keys())
  #for pred_id, sense in predicates.iteritems():
  for pred_id in sorted_pred_ids:
    assert pred_id < len(tokens)
    print ('*{} ({})'.format(tokens[pred_id], pred_id))
    assert pred_id in arguments
    for arg_start, arg_end, role in arguments[pred_id]:
      if role == 'V':
        continue
      if not span_is_exclusive:
        arg_end += 1
      print ('\t{}: {} [{}-{})'.format(
          role, ' '.join(tokens[arg_start:arg_end]),
          arg_start, arg_end))


def ReadFromCoNLL(conll_input_path):
  """Read predicates and arguments information from CoNLL-formatted
     (predicate-arguments only) files.

  Returns:
    all_predicates:
    all_arguments:
  """
  all_predicates = [[]]  # Lists of predicates including the dummy '-'.
  all_arguments = [{}]  # From pred_id to list of argument tuples.
  args_buffer = []
  token_count = 0
  with open(conll_input_path, 'r') as conll_input:
    for line in conll_input:
      line = line.strip()
      if not line:  # Sentence break.
        # Collect all arguments.
        pred_counter = 0
        for pred_id, pred_info in enumerate(all_predicates[-1]):
          if pred_info == '-':
            continue
          for j, s in enumerate(args_buffer[pred_counter]):
            if '(' in s:
              role = s.strip('()*')
              all_arguments[-1][pred_id].append([j, -1, role])  # Start, end, label.
            if ')' in s:
              all_arguments[-1][pred_id][-1][1] = j  # Change span end index.
          pred_counter += 1
        # 
        all_predicates.append([])
        all_arguments.append({})
        args_buffer = []
        token_count = 0
      elif line:
        line_info = line.split()
        all_predicates[-1].append(line_info[0])
        if line_info[0] != '-':
          # Add current predicate ID.
          all_arguments[-1][token_count] = []
        token_count += 1
        # Buffer current argument information.
        line_info = line_info[1:]
        if not args_buffer:
          args_buffer = [[] for _ in line_info]
        for i, s in enumerate(line_info):
          args_buffer[i].append(s)
    conll_input.close()
  return all_predicates, all_arguments


def PrintToCoNLL(all_tokens, all_predicates, all_arguments, conll_output_path,
                 gold_conll_path=None):
  """ Print predicates and arguments to CoNLL format.
      If gold_conll_path is provided, use gold predicate senses.
  """
  gold_predicates = [[]]
  # Read gold predicates if gold_conll_path is provided.
  if gold_conll_path:
    with open(gold_conll_path, 'r') as gold_conll_input:
      for line in gold_conll_input:
        line = line.strip()
        if not line:  # Sentence break.
          gold_predicates.append([])
        elif line:
          gold_predicates[-1].append(line.split()[0])
      if not gold_predicates[-1]:
        gold_predicates.pop()
    print ('Read gold predicates from {} sentences.'.format(len(gold_predicates)))
    assert len(gold_predicates) == len(all_predicates)
    gold_conll_input.close()

  # Print SRL prediction to conll.
  with open(conll_output_path, 'w') as conll_output:
    for sent_id, (sentence, preds, args) in enumerate(
        zip(all_tokens, all_predicates, all_arguments)):
      gold_preds = gold_predicates[sent_id] if gold_predicates else None
      slen = len(sentence)
      pred_column = ['-' for _ in sentence]
      arg_columns = [[] for _ in sentence]
      for pred_id, sense in preds.iteritems():
        if gold_preds and gold_preds[pred_id] != '-':
          # Use gold predicate lemma for eval purpose.
          pred_column[pred_id] = gold_preds[pred_id]
        else:
          # Will be counted as false positive.
          # So doesn't really matter what we put here.
          pred_column[pred_id] = 'P' + all_tokens[sent_id][pred_id]
        # Initialize argument columns.
        arg_columns[pred_id] = ['*' for _  in sentence]
        arg_columns[pred_id][pred_id] = '(V*)'

      for pred_id, p_args in args.iteritems():
        for start, end, role in p_args:  # Spans are exclusive.
          arg_columns[pred_id][start] = '(' + role + arg_columns[pred_id][start]
          arg_columns[pred_id][end-1] = arg_columns[pred_id][end-1] + ')'

      # Print all columns.
      arg_columns = [c for c in arg_columns if c]
      for i in range(len(sentence)):
        conll_output.write('{0:<15}'.format(pred_column[i]))
        for col in arg_columns:
          conll_output.write('\t{0:>12}'.format(col[i]))
        conll_output.write('\n')
      # Blank line for sentence break.
      conll_output.write('\n')
    conll_output.close()

 

