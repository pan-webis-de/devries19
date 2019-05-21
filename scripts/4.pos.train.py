import os

treebankPrefix = {'en': 'ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-', 'fr':'ud-treebanks-v2.3/UD_French-GSD/fr_gsd-ud-', 'it':'ud-treebanks-v2.3/UD_Italian-ISDT/it_isdt-ud-', 'sp':'ud-treebanks-v2.3/UD_Spanish-GSD/es_gsd-ud-'}
    
def genPosData(inFile, outFile):
    cmd = 'grep -v "^#" ' + inFile + ' | cut -f 2,4 > ' + outFile
    os.system(cmd)

if not os.path.exists('models'):
    os.mkdir('models')

for lang in treebankPrefix:
    print('Train POS tagger for {}'.format(lang))

    trainDep = treebankPrefix[lang] + 'train.conllu'
    trainPos = treebankPrefix[lang] + 'train.pos'
    genPosData(trainDep, trainPos)
    devDep = treebankPrefix[lang] + 'dev.conllu'
    devPos = treebankPrefix[lang] + 'dev.pos'
    genPosData(devDep, devPos)

    cmd = 'cd bilstm-aux && python3 src/structbilty.py --train ../' + trainPos + ' --dev ../' + devPos + ' --model ../models/' + lang + 'Pos && cd ..'
    
    print(cmd)
    # os.system(cmd)

