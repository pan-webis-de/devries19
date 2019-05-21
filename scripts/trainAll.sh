./scripts/1.tokenize.prep.sh

# python3 scripts/2.elmo.prep.py
# python3 scripts/2.elmo.writeEmb.train.py

./scripts/3.depParse.prep.sh
# python3 scripts/3.depParse.train.py
python3 3.depParse.train.woEmbeds.py

./scripts/4.pos.prep.sh
python3 scripts/4.pos.train.py

#echo "python3 scripts/5.const.parse.py" > 5.run.sh
#python3 scripts/pg.prep.py 5.run.sh 5.run 10 4 2 1
#python3 scripts/pg.run.py 5.run.1
