import json
from glob import glob
import sys
from elmoformanylangs import Embedder

input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training-dataset-2019-01-23'

with open('{}/collection-info.json'.format(input_path), 'r') as f:
    collectioninfo = json.load(f)


converters = {
    'en': Embedder('ELMoForManyLangs/en'),
    'fr': Embedder('ELMoForManyLangs/fr'),
    'sp': Embedder('ELMoForManyLangs/es'),
    'it': Embedder('ELMoForManyLangs/it')
}


def embed(inFile, converter):
    curTree = []
    trees = []

    # Read sentence trees
    for line in open(inFile):
        if len(line) < 2:
            trees.append(curTree)
            curTree = []
        elif line[0] != '#':
            tok = line.strip().split('\t')
            curTree.append(tok)

    # Embed all sentences
    sents = [[x[1] for x in curTree] for curTree in trees]
    embeddings = converter.sents2elmo(sents)

    # Add embeddings to trees
    outFile = open(inFile + '.elmo', 'w')
    for curTree, emb in zip(trees, embeddings):
        for itemIdx in range(len(curTree)):
            embStr = 'emb=' + ','.join([str(x) for x in emb[itemIdx]])
            if curTree[itemIdx][-1] == '_':
                curTree[itemIdx][-1] = embStr
            else:
                curTree[itemIdx][-1] += '|' + embStr
            outFile.write('\t'.join(curTree[itemIdx]) + '\n')
        outFile.write('\n')

    outFile.close()


for probleminfo in collectioninfo:
    problem_name = probleminfo['problem-name']
    lang = probleminfo['language']

    print('Embedding {}'.format(problem_name))

    for inFile in glob('{}/{}/*/*.tok'.format(input_path, problem_name)):
        embed(inFile, converters[lang])
