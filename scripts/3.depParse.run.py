import pickle
import os
import json
from glob import glob
import sys
import codecs

from uuparserutils.utils import Treebank, ConllEntry
from uuparserutils.arc_hybrid import ArcHybridLSTM as Parser


def load_parser(modeldir):
    params = os.path.join(modeldir, 'params.pickle')

    print('Reading params from ' + params)
    with open(params, 'r') as paramsfp:
        stored_vocab, stored_opt = pickle.load(paramsfp)

        parser = Parser(stored_vocab, stored_opt)
        model = os.path.join(modeldir, 'barchybrid.model')
        parser.Load(model)
        return parser, stored_opt


use_embeddings = len(sys.argv) > 2 and sys.argv[2] == 'embedding'


print 'Loading parsers'
parsers = {
    'en': load_parser('models/en{}'.format('' if use_embeddings else '.woEmbeds')),
    'fr': load_parser('models/fr{}'.format('' if use_embeddings else '.woEmbeds')),
    'it': load_parser('models/it{}'.format('' if use_embeddings else '.woEmbeds')),
    'sp': load_parser('models/sp{}'.format('' if use_embeddings else '.woEmbeds')),
}
print 'Finished loading parsers'


def write_conll(conll_gen, treebanks):
    tbank_dict = {treebank.iso_id: treebank for treebank in treebanks}
    cur_tbank = conll_gen[0][0].treebank_id
    outfile = tbank_dict[cur_tbank].outfilename
    fh = codecs.open(outfile, 'w', encoding='utf-8')
    print "Writing to " + outfile
    for sentence in conll_gen:
        if cur_tbank != sentence[0].treebank_id:
            fh.close()
            cur_tbank = sentence[0].treebank_id
            outfile = tbank_dict[cur_tbank].outfilename
            fh = codecs.open(outfile, 'w', encoding='utf-8')
            print "Writing to " + outfile
        for entry in sentence[1:]:
            if not isinstance(entry, ConllEntry):
                continue

            entry.misc = None
            fh.write(unicode(entry) + '\n')
        fh.write('\n')


def run(parser, stored_opt, treebanks):
    pred = list(parser.Predict(treebanks, "test", stored_opt))
    write_conll(pred, treebanks)


input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training-dataset-2019-01-23'

with open('{}/collection-info.json'.format(input_path), 'r') as f:
    collectioninfo = json.load(f)


for probleminfo in collectioninfo:
    problem_name = probleminfo['problem-name']
    lang = probleminfo['language']

    print 'Parsing ' + problem_name

    parser, stored_opt = parsers[lang]

    for inFile in glob('{}/{}/*/*.tok{}'.format(input_path, problem_name, '.elmo' if use_embeddings else '')):
        outFile = inFile.replace('tok.elmo', 'dep') if use_embeddings else inFile.replace('tok', 'dep')

        treebank = Treebank(None, None, inFile)
        treebank.iso_id = None
        treebank.outfilename = outFile
        treebank.modeldir = 'models/{}{}'.format(lang, '' if use_embeddings else '.woEmbeds')

        run(parser, stored_opt, [treebank])
