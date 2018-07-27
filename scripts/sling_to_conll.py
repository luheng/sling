"""Convert Sling records to CoNLL format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from collections import Iterable
import sling
import subprocess

FLAGS = flags.FLAGS

# TODO: Remove later.
_GOLD_CONLL_FILE = './ontonotes_data/v5.dev.props.gold.txt'
_SRL_CONLL_EVAL_SCRIPT  = "./scripts/run_conll_eval.sh"


def RetrieveTokens(doc):
  tokens = []
  for token in doc.tokens:
    tokens.append(token.text)
  #print (tokens)
  return tokens


def RetrieveSRL(doc, commons):
  """ Recover SRL predicate-argument annotations from Sling format.
  """
  frame_to_id = {}
  id_to_span = {}
  isa = commons["isa"]
  count = 0
  for mention in doc.mentions:
    for frame in mention.evokes():
      frame_to_id[frame] = count
      id_to_span[count] = (mention.begin, mention.end)
      count += 1

  predicates = {}  # Map from predicate to sense (frame), e.g. /pb/sense.01.
  arguments = {}  # Map from predicate to list of (arg_start, arg_end, lable) tuples.
  # Process all the predicates.
  for mention in doc.mentions:
    for frame in mention.evokes():
      frame_type = str(frame[isa])
      if frame_type.startswith('/pb/predicate'):
        predicate_id = mention.begin
        predicates[predicate_id] = frame_type.split('/')[-1]
        arguments[predicate_id] = []
        # print ("Predicate", predicate_id, frame_type)

  for mention in doc.mentions:
    start = mention.begin
    end = mention.end  # exclusive
    for frame in mention.evokes():
      frame_id = frame_to_id[frame]
      frame_type = str(frame[isa])
      # print ("Span", frame_id, start, end, frame_type)
      if frame_type.startswith('/pb/predicate'):
        for slot, value in frame:
          if slot != isa:
            assert value in frame_to_id
            role = str(slot).split('/')[-1]
            # print ("Argument", frame_to_id[value], slot)
            arg_start, arg_end = id_to_span[frame_to_id[value]]
            arguments[start].append((arg_start, arg_end, role))

  assert len(predicates) == len(arguments)
  return predicates, arguments


def PrintHumanReadable(tokens, predicates, arguments):
  """ Print SRL predictions into human-readable format.
  """
  # Print the sentence.
  print (' '.join(tokens))
  # Print predicate-level information.
  for pred_id, sense in predicates.iteritems():
    assert pred_id < len(tokens)
    print ('{}: {}'.format(sense, tokens[pred_id]))
    assert pred_id in arguments
    for arg_start, arg_end, role in arguments[pred_id]:
      print ('\t{}: {}'.format(role, ' '.join(tokens[arg_start:arg_end])))
  print ('\n')


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
          pred_column[pred_id] = sense
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

 
def SlingToCoNLL(sling_path, gold_conll_path, conll_output_path):
  """ We need gold CoNLL file to align the predicates. (?)
  """
  commons = sling.Store()
  commons.load("./ontonotes_data/ontonotes_v5_commons")
  schema = sling.DocumentSchema(commons)
  commons.freeze()
  reader = sling.RecordReader(sling_path)
  all_tokens = []
  all_predicates = []
  all_arguments = []

  for key, val in reader:
    store = sling.Store(commons)
    sling_doc = sling.Document(store.parse(val), schema=schema)
    tokens = RetrieveTokens(sling_doc)
    #print (sling_doc.frame.data(binary=False), '\n')
    # print (sling_doc.frame['/s/document/tokens'])
    predicates, arguments = RetrieveSRL(sling_doc, commons)
    #PrintHumanReadable(tokens, predicates, arguments)
    all_tokens.append(tokens)
    all_predicates.append(predicates)
    all_arguments.append(arguments)

  print (len(all_tokens), len(all_predicates), len(all_arguments))
  PrintToCoNLL(all_tokens, all_predicates, all_arguments,
               conll_output_path, gold_conll_path)
  reader.close()


def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')
  sling_prediction_file = argv[1]
  temp_output = '/tmp/sling.out.conll'
  SlingToCoNLL(sling_prediction_file, _GOLD_CONLL_FILE, temp_output)

  # Evalute twice with official script.
  child = subprocess.Popen(
      'sh {} {} {}'.format(_SRL_CONLL_EVAL_SCRIPT, _GOLD_CONLL_FILE, temp_output),
      shell=True, stdout=subprocess.PIPE)
  eval_info = child.communicate()[0]
  child2 = subprocess.Popen(
      'sh {} {} {}'.format(_SRL_CONLL_EVAL_SCRIPT, temp_output, _GOLD_CONLL_FILE),
      shell=True, stdout=subprocess.PIPE)
  eval_info2 = child2.communicate()[0]
  try:
    conll_recall = float(eval_info.strip().split('\n')[6].strip().split()[5])
    conll_precision = float(eval_info2.strip().split('\n')[6].strip().split()[5])
    if conll_recall + conll_precision > 0:
      conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
    else:
      conll_f1 = 0
    print(eval_info)
    print(eval_info2)
    print("Official CoNLL Precision:\t{}\tRecall:\t{}\tFscore:\t{}".format(
        conll_precision, conll_recall, conll_f1))
  except IndexError:
    conll_recall = 0
    conll_precision = 0
    conll_f1 = 0
    print("Unable to get FScore. Skipping.")


if __name__ == '__main__':
  app.run(main)
