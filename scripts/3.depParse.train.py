import os

treebankPrefix = {
    'en': 'ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-',
    'fr': 'ud-treebanks-v2.3/UD_French-GSD/fr_gsd-ud-',
    'it': 'ud-treebanks-v2.3/UD_Italian-ISDT/it_isdt-ud-',
    'sp': 'ud-treebanks-v2.3/UD_Spanish-GSD/es_gsd-ud-'
}

if not os.path.exists('models'):
    os.mkdir('models')

for lang in treebankPrefix:
    train = '../../' + treebankPrefix[lang] + 'train.conllu.elmo'
    dev = '../../' + treebankPrefix[lang] + 'dev.conllu.elmo'
    cmd = 'cd uuparser/barchybrid && python2 src/parser.py --trainfile ' + \
        train + ' --devfile ' + dev + ' --outdir ../../models/' + lang
    print(cmd)
