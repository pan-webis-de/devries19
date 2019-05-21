import sys
from elmoformanylangs import Embedder

if len(sys.argv) < 3:
    print('please provide embeddings and conl file')
    exit(0)

converter = Embedder(sys.argv[1])
curComments = ''
curTree = []
outFile = open(sys.argv[2] + '.elmo', 'w')
for line in open(sys.argv[2]):
    if len(line) < 2:
        outFile.write(curComments.strip() + '\n')
        sent = [[x[1] for x in curTree]]
        emb = converter.sents2elmo(sent)[0]
        for itemIdx in range(len(curTree)):
            embStr = 'emb=' + ','.join([str(x) for x in emb[itemIdx]])
            if curTree[itemIdx][-1] == '_':
                curTree[itemIdx][-1] = embStr
            else:
                curTree[itemIdx][-1] += '|' + embStr
            outFile.write('\t'.join(curTree[itemIdx]) + '\n')
        outFile.write('\n')
        curComments = ''
        curTree = []
    elif line[0] == '#':
        curComments += line
    else:
        tok = line.strip().split('\t')
        curTree.append(tok)

outFile.close()
