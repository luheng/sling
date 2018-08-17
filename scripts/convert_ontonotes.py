"""Converting OntoNotes SRL data from json to SLING format and gold CoNLL format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import json
import sling

FLAGS = flags.FLAGS

flags.DEFINE_string('train_input',
                    'ontonotes_data/train.english.mtl.jsonlines',
                    'Json-formatted training data.')

flags.DEFINE_string('dev_input',
                    'ontonotes_data/dev.english.mtl.jsonlines',
                    'Json-formatted development data.')

flags.DEFINE_string('test_input',
                    'ontonotes_data/test.english.mtl.jsonlines',
                    'Json-formatted test data.')

flags.DEFINE_boolean('remove_v_args', 1,
                     'Remove V-V connections (self-loop) in training data')

flags.DEFINE_string('train_output',
                    'ontonotes_data/v5.train.rec',
                    'SLING-formatted training data after conversion.')

flags.DEFINE_string('dev_output',
                    'ontonotes_data/v5.dev.rec',
                    'SLING-formatted development data after conversion.')

flags.DEFINE_string('test_output',
                    'ontonotes_data/v5.test.rec',
                    'SLING-formatted test data after conversion.')

flags.DEFINE_string('dev_conll_output',
                    'ontonotes_data/v5.dev.props.gold.txt',
                    'CoNLL-formatted development for official evaluation.')

flags.DEFINE_string('test_conll_output',
                    'ontonotes_data/v5.test.props.gold.txt',
                    'CoNLL-formatted test data for official evaluation.')

flags.DEFINE_string('base_commons_path',
                    '/tmp/commons',
                    'Path to the original commons file.')

flags.DEFINE_string('commons_path',
                    'ontonotes_data/ontonotes_v5_commons',
                    'Path to the updated commons file.')


def CreateCommons(base_commons_path, commons_path, input_path):
  '''Generate the frame inventory.
  '''
  # Create commons store.
  commons = sling.Store()
  commons.load(base_commons_path)  # Basically the frame inventory.
  with open(input_path, 'r') as json_input:
    for line in json_input.readlines():
      doc_info = json.loads(line.strip())
      sentences = doc_info['sentences']
      # We do not model predicate sense for now.
      # for pred_info in doc_info['predicates']:
      #  for pred in pred_info:
      #    if pred != '-':
      #      frame = str('.'.join(pred.split('.')[:-1]))
      #      _ = commons.frame({'id': '/pb/'+frame})
      for srl_info in doc_info['srl']:
        for pred_id, arg_start, arg_end, role in srl_info:
          _ = commons.frame({'id': '/pb/'+str(role)})

  #_ = commons.frame({'id': '/saft/arg'})
  _ = commons.frame({'id': '/pb/argument'})  # Generic argument types.
  _ = commons.frame({'id': '/pb/predicate'})  # Generic predicate types.
  commons.freeze()
  commons.save(commons_path, binary=True)


def AddTokensToDocument(sling_doc, sentences):
  '''Create a Sling document using tokens from the json object.
  '''
  char_offset = 0
  for sentence in sentences:
    for i, token in enumerate(sentence):
      token = token.encode('utf-8')
      tlen = len(token)
      if i == 0:
        sling_doc.add_token(token, start=char_offset, length=tlen, brk=0)  # SENTENCE_BREAK
      else:
        sling_doc.add_token(token, start=char_offset, length=tlen, brk=1)  # SPACE_BREAK
      char_offset += tlen
  return sling_doc


def ConvertData(input_path, output_path, commons_path, conll_output_path=None):
  '''Convert jsonlines data to Sling format.
  '''
  commons = sling.Store()
  commons.load(commons_path)
  doc_schema = sling.DocumentSchema(commons)
  commons.freeze()
  isa = commons['isa']

  sentence_count = 0
  token_count = 0
  predicate_count = 0
  pas_count = 0  # Total number of predicate-argument relations.
  empty_sentence_count = 0

  output_writer = sling.RecordWriter(output_path)
  conll_writer = open(conll_output_path, 'w') if conll_output_path else None

  with open(input_path, 'r') as json_input:
    for line in json_input.readlines():
      doc_info = json.loads(line.strip())
      # Make everything sentence-level for now.
      token_id_offset = 0
      for sent_id, sentence in enumerate(doc_info['sentences']):
        srl_info = doc_info['srl'][sent_id]
        if 'predicates' in doc_info:
          predicates = doc_info['predicates'][sent_id]
          num_predicates = len([p for p in predicates if p != '-'])
        else:
          pred_set = set([p[0]-token_id_offset for p in srl_info])
          predicates = [token if i in pred_set else '-' for (
              i, token) in enumerate(sentence)]
          num_predicates = len(pred_set)

        if not sentence:
          empty_sentence_count += 1
          continue

        sentence_count += 1
        token_count += len(sentence)
        predicate_count += num_predicates
        pas_count += len(srl_info)

        store = sling.Store(commons)
        sling_doc = sling.Document(store=store,schema=doc_schema)
        AddTokensToDocument(sling_doc, [sentence,])

        args_buf = []  # For debugging.
        target_frames = [None for _ in sentence]
        for i, frame in enumerate(predicates):
          if frame != '-':
            #frame = '.'.join(str(frame).split('.')[:-1])
            #target_frames[i] = store.frame({isa: commons['/pb/'+frame]})
            target_frames[i] = store.frame({isa: commons['/pb/predicate']})
            # Spans are exclusive!
            sling_doc.add_mention(i, i+1).evoke(target_frames[i])

        # Write SRL info to SLING record.
        for pred_id, arg_start, arg_end, role in srl_info:
          role = str(role)
          if FLAGS.remove_v_args and role == 'V':
            print ('Removed V args')
            continue
          # Add argument evoke to default /saft/arg or /pb/argument
          arg_frame = store.frame({isa: commons['/pb/argument']})
          # Spans are exclusive!!
          sling_doc.add_mention(
              arg_start-token_id_offset, arg_end-token_id_offset+1
          ).evoke(arg_frame)
          # Add predicate-argument link.
          target_frames[pred_id-token_id_offset].append(
              commons['/pb/'+role], arg_frame)
        sling_doc.update()
        output_writer.write(doc_info['doc_key']+'.'+str(sent_id),
                            sling_doc.frame.data(binary=True))

        # WRite SRL info to CoNLL format.
        if conll_writer:
          # Each column correspond to a token.
          columns = []
          for pid, pred in enumerate(predicates):
            if pred != '-':
              columns.append(['*' for _ in predicates])
              #columns[-1][pid] = '(V*)'
            else:
              columns.append([])
          for pred_id, arg_start, arg_end, role in srl_info:
            pid, sid, eid = (
                pred_id-token_id_offset, arg_start-token_id_offset,
                arg_end-token_id_offset)
            columns[pid][sid] = '(' + role + columns[pid][sid]
            columns[pid][eid] = columns[pid][eid] + ')'
          columns = [c for c in columns if c]
          for pid, pred in enumerate(predicates):
            conll_writer.write('{0:<15}'.format(pred))
            for c in columns:
              conll_writer.write('\t{0:>12}'.format(c[pid]))
            conll_writer.write('\n')
          # One blank line between each sentence.
          conll_writer.write('\n')

        # Update token_id_offset to convert positions to sentence-level.
        token_id_offset += len(sentence)

  output_writer.close()
  if conll_writer:
    conll_writer.close()
  print ('Wrote {} sentences (skipping {} empty ones), to {}'.format(
      sentence_count, empty_sentence_count, output_path))
  print ('Split contains {} tokens, {} predicates, and {} PAS tuples (counting V args).'.format(
      token_count, predicate_count, pas_count))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  CreateCommons(FLAGS.base_commons_path, FLAGS.commons_path, FLAGS.train_input)
  ConvertData(FLAGS.train_input, FLAGS.train_output, FLAGS.commons_path, None)
  ConvertData(FLAGS.dev_input, FLAGS.dev_output, FLAGS.commons_path,
              FLAGS.dev_conll_output)
  ConvertData(FLAGS.test_input, FLAGS.test_output, FLAGS.commons_path,
              FLAGS.test_conll_output)


if __name__ == '__main__':
  app.run(main)

