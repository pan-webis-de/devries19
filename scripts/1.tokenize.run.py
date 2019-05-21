import json
from glob import glob
import sys
from ufal.udpipe import Model, Pipeline, ProcessingError

input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training-dataset-2019-01-23'

print('Loading UDPipe models')
models = {
    'en':  Model.load('udpipe/src/udpipe-ud-2.3-181115/english-ewt-ud-2.3-181115.udpipe'),
    'fr':  Model.load('udpipe/src/udpipe-ud-2.3-181115/french-gsd-ud-2.3-181115.udpipe'),
    'it':  Model.load('udpipe/src/udpipe-ud-2.3-181115/italian-isdt-ud-2.3-181115.udpipe'),
    'sp':  Model.load('udpipe/src/udpipe-ud-2.3-181115/spanish-gsd-ud-2.3-181115.udpipe')
}
print('Finished loading UDPipe models')

if not models['en'] or not models['fr'] or not models['it'] or not models['sp']:
    print('Cannot load one of the models')
    sys.exit(1)

pipelines = {
    'en':  Pipeline(models['en'], 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu'),
    'fr':  Pipeline(models['fr'], 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu'),
    'it':  Pipeline(models['it'], 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu'),
    'sp':  Pipeline(models['sp'], 'tokenize', Pipeline.NONE, Pipeline.NONE, 'conllu')
}
error = ProcessingError()

with open('{}/collection-info.json'.format(input_path), 'r') as f:
    collectioninfo = json.load(f)

for probleminfo in collectioninfo:
    problem_name = probleminfo['problem-name']

    print('Tokenizing {}'.format(problem_name))

    for inFile in glob('{}/{}/*/*.seq'.format(input_path, problem_name)):
        outFile = inFile.replace('seq', 'tok')

        with open(inFile) as f:
            raw_text = ''.join(f.readlines())

        processed_text = pipelines[probleminfo['language']].process(raw_text, error)
        if error.occurred():
            print('UDPipe error: {}'.format(error.message))
            exit(1)

        with open(outFile, 'w') as f:
            f.write(processed_text)
