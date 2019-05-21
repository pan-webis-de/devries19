import benepar
import json
import os
from glob import glob
import sys

input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training-dataset-2019-01-23'

enParser = benepar.Parser("benepar_en2")
frParser = benepar.Parser("benepar_fr")

with open('{}/collection-info.json'.format(input_path), 'r') as f:
    collectioninfo = json.load(f)

for probleminfo in collectioninfo:
    problem_name = probleminfo['problem-name']
    language = probleminfo['language']
    if language == 'sp' or language == 'it':
        continue
    for inFile in glob('{}/{}/*/*.rawTok'.format(input_path, problem_name)):
        outFile = open(inFile.replace('rawTok', 'const'), 'w')
        print(outFile)
        for sent in open(inFile):
            if language == 'en':
                tree = enParser.parse(sent)
            if language == 'fr':
                tree = frParser.parse(sent)
            outFile.write(' '.join(str(tree).split()) + '\n')
        outFile.close()
    

