for inFile in ${1:-data/training-dataset-2019-01-23}/problem*/*/*tok;
do
    outFile="${inFile%.*}".rawTok
    grep -v "^#" $inFile | cut -f 2 | tr '\n' ' ' | sed 's/  /\'$'\n''/g' > $outFile
done
