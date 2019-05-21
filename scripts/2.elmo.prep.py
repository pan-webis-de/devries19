import os
import json

os.system('git clone https://github.com/HIT-SCIR/ELMoForManyLangs.git')
os.system('pip3 ELMoForManyLangs/setup.py install')

langs = ['en', 'es', 'fr', 'it']
numbers = ['144', '145', '150', '159']

for lang, number in zip(langs, numbers):
    embDir = 'ELMoForManyLangs/' + lang + '/'
    os.system('wget http://vectors.nlpl.eu/repository/11/' + number + '.zip')
    if not os.path.exists(embDir):
        os.mkdir(embDir)
    os.rename(number + '.zip', embDir + number + '.zip')
    os.system('cd ' + embDir + ' && unzip ' + number + '.zip && rm ' + number + '.zip && cd ../../')

for lang in langs:
    data = json.load(open('ELMoForManyLangs/' + lang + '/config.json'))
    data['config_path'] = '../configs/cnn_50_100_512_4096_sample.json'
    json.dump(data, open('ELMoForManyLangs/' + lang + '/config.json', 'w'))
