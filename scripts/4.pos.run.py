import json
from glob import glob
import sys
import random
import numpy as np
from bilstmauxutils.lib.mnnl import init_dynet
from bilstmauxutils.lib.mio import SeqData
from bilstmauxutils.structbilty import load
import codecs

input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training-dataset-2019-01-23'


seed = random.randint(1, 99999999)
np.random.seed(seed)
random.seed(seed)

init_dynet(seed)

print('Loading taggers')
taggers = {
    'en': load('models/enPos', None),
    'fr': load('models/frPos', None),
    'it': load('models/itPos', None),
    'sp': load('models/spPos', None),
}
print('Finished loading taggers')


with open('{}/collection-info.json'.format(input_path), 'r') as f:
    collectioninfo = json.load(f)


for probleminfo in collectioninfo:
    problem_name = probleminfo['problem-name']
    lang = probleminfo['language']

    terminal_out = sys.stdout

    print('POS tagging {}'.format(problem_name))

    tagger = taggers[lang]

    input_files = list(glob('{}/{}/*/*.rawTok'.format(input_path, problem_name)))
    for i, in_file in enumerate(input_files):
        if i % (int(len(input_files) / 50) + 1) == 0:
            print('{}/{} ({}%)'.format(i, len(input_files), round(i/len(input_files)*100)))

        test = SeqData([in_file], raw=True)
        out_file = in_file.replace('rawTok', 'pos')

        sys.stdout = codecs.open(out_file + 'task0', 'w', encoding='utf-8')
        correct, total = tagger.evaluate(test, 'task0',
                                         output_predictions=out_file,
                                         output_confidences=False, raw=True,
                                         unk_tag=None)
        sys.stdout = terminal_out
