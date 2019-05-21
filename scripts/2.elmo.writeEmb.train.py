import json
import os
from glob import glob

with open('data/training-dataset-2019-01-23/collection-info.json', 'r') as f:
    collectioninfo = json.load(f)

treebanks = {'en': 'ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-', 'fr':'ud-treebanks-v2.3/UD_French-GSD/fr_gsd-ud-', 'it':'ud-treebanks-v2.3/UD_Italian-ISDT/it_isdt-ud-', 'es':'ud-treebanks-v2.3/UD_Spanish-GSD/es_gsd-ud-'}

for lang in treebanks:
    dataset = treebanks[lang]
    for datasplit in ['train', 'dev']:
        inFile = dataset + datasplit + '.conllu'
        cmd = 'cd ELMoForManyLangs'
        cmd += ' && python3 2.elmo.py ' + lang + ' ../' + inFile 
        cmd += ' && cd ../'
        print(cmd)

