from nltk.tokenize import sent_tokenize
import json
from glob import glob
import sys


input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training-dataset-2019-01-23'

with open('{}/collection-info.json'.format(input_path), 'r') as f:
    collectioninfo = json.load(f)

langmap = {
    'en': 'english',
    'fr': 'french',
    'it': 'italian',
    'sp': 'spanish'
}

for probleminfo in collectioninfo:
    problem_name = probleminfo['problem-name']
    lang = langmap[probleminfo['language']]

    print('Sentence tokenizing {}'.format(problem_name))

    for filename in glob('{}/{}/*/*.txt'.format(input_path, problem_name)):
        with open(filename, 'r') as f:
            sents = []
            for line in f.readlines():
                sents = sents + sent_tokenize(line.rstrip(), lang)

        with open(filename + '.seq', 'w') as f:
            f.write('\n'.join(sents))
