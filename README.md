# Cross-domain Authorship Attribution

A support vector machine based approach for cross-domain authorship attribution. This application is specifically created for the [PAN19 cross-domain authorship attribution](https://pan.webis.de/clef19/pan19-web/author-identification.html) shared task.

## Requirements
 - Both Python 3.6 and Python 2.7 (Python 2.7 is needed for the [dependency parser](https://github.com/UppsalaNLP/uuparser))
 - Python3 packages in `python3-requirements.txt`
 - Python2 packages in `python2-requirements.txt`
 - An official data set that is compiled for this shared task or at least in the same format

## Installation
Before the application can be used, some external data is needed and the part-of-speech tagger and dependency parser have to be trained. To do this, run the following script:
    
    ./scripts/trainAll.sh

**Warning**: This will take a while.

## Usage
Before the SVM model can be used for training and testing, the input data must be preprocessed once. This can be done by running the following script where the second path must be to an non-existing directory:

    ./scripts/runAll.sh path/to/training-dataset-2019-01-23 path/to/training-dataset-2019-01-23-processed

The preprocessing can also be triggered by adding the `-r` argument to the main script:

    python3 svm.py -r data/training-dataset-2019-01-23 -i data/training-dataset-2019-01-23-processed -o outputs

If the input data is already processed you should run the script without the `-r` argument:

    python3 svm.py -i data/training-dataset-2019-01-23-processed -o outputs

If you want to want to see macro-averaged precision, recall and f-scores during processing, add the `--eval` argument. The values are identical to the official shared task metrics:

    python3 svm.py -i data/training-dataset-2019-01-23-processed -o outputs --eval