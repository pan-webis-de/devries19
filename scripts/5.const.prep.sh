pip3 install --user cython numpy
pip3 install --user benepar[cpu]

echo -e "import nltk\nnltk.download('punkt')\n" | python3

echo -e "import benepar\nbenepar.download('benepar_en2')" | python3
echo -e "import benepar\nbenepar.download('benepar_fr')" | python3



