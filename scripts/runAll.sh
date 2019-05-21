cp -r $1 $2

python3 scripts/0.sent_tokenize.run.py $2

python3 scripts/1.tokenize.run.py $2

python3 scripts/2.elmo.writeEmb.py $2

python2 scripts/3.depParse.run.py $2 embedding

./scripts/4.pos.run.sh $2
python3 scripts/4.pos.run.py $2

for f in $2/problem*/*/*postask0; do mv $f `echo $f | sed 's/postask0/pos/'` ; done