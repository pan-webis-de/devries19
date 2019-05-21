"""
Copyright 2019 Wietse de Vries and Martijn Bartelds

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import json
from glob import glob
import os
import time
import argparse
import pickle
import codecs
import warnings

import pandas as pd
import scipy as sp
from conll_df import conll_df

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics.classification import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

FLOAT_DTYPES = (np.float64, np.float32, np.float16)


def read_problem_info(path, problem_name):
    problem_path = path + os.sep + problem_name
    probleminfo_path = problem_path + os.sep + 'problem-info.json'

    with open(probleminfo_path, 'r') as f:
        probleminfo = json.load(f)

    def list_files(candidate): return glob(problem_path + os.sep + candidate + os.sep + '*.txt')

    candidates = {c['author-name']: list_files(c['author-name']) for c in probleminfo['candidate-authors']}
    return candidates, list_files(probleminfo['unknown-folder'])


def read_collection_info(path):
    with open(path + os.sep + 'collection-info.json', 'r') as f:
        collectioninfo = json.load(f)

    for probleminfo in collectioninfo:
        candidates, unk_folder = read_problem_info(path, probleminfo['problem-name'])
        probleminfo['candidates'] = candidates
        probleminfo['unknown'] = unk_folder

    return collectioninfo


def save_output(outpath, problem_name, test_names, predictions):
    os.makedirs(outpath, exist_ok=True)
    out_data = []

    for unknown, prediction in zip(test_names, predictions):
        unknown_name = unknown.split('/')[-1]
        out_data.append(
            {'unknown-text': unknown_name, 'predicted-author': prediction})

    with open(outpath + os.sep + 'answers-' + problem_name + '.json', 'w') as f:
        json.dump(out_data, f, indent=4)

    print(' > answers saved to file', 'answers-' + problem_name + '.json')


def get_model_inputs(candidates):
    filenames, labels = [], []
    for c, c_files in candidates.items():
        filenames.extend(c_files)
        labels.extend([c] * len(c_files))

    return filenames, labels


def get_test_labels(filenames, collectioninfo, path):
    filenames = [f.replace(path + os.sep, '').split(os.sep) for f in filenames]
    labels = []
    truths = {}
    for problem, _, filename in filenames:
        if problem not in truths:
            with open(path + os.sep + problem + '/ground-truth.json', 'r') as f:
                truth = json.load(f)['ground_truth']
                truths[problem] = {t['unknown-text']: t['true-author'] for t in truth}
        labels.append(truths[problem][filename])

    return labels


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ Based on https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html """

    def fit(self, x, y=None):
        return self

    def _read_text(self, filename):
        """ Read raw texts from .txt file """
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_words(self, filename):
        """ Read tokens and pos tags from .txt.pos file """
        with codecs.open(filename + '.pos', 'r', encoding='utf-8') as f:
            tokens, pos_tags = [], []
            for line in f.readlines():
                word = line.rstrip().split()
                if len(word) == 2:
                    tokens.append(word[0])
                    pos_tags.append(word[1])
            return tokens, pos_tags

    def _read_syntax(self, filename):
        dep_df = conll_df(filename + '.dep', file_index=False)
        if dep_df.index.levels[1].dtype != np.int64:
            dep_df.reset_index(inplace=True)
            dep_df = dep_df[dep_df['i'].str.isnumeric()]
            dep_df['i'] = dep_df['i'].astype(int)
            dep_df.set_index(['s', 'i'], inplace=True)
        return dep_df.rename(columns={'w': 'word', 'l': 'lemma', 'f': 'dep'})

    def transform(self, filenames):
        features = np.recarray(shape=(len(filenames),),
                               dtype=[
                                   ('text', object),
                                   ('tokens', object),
                                   ('pos_tags', object),
                                   ('syntax', object)
        ])
        for i, filename in enumerate(filenames):
            features['text'][i] = self._read_text(filename)
            features['tokens'][i], features['pos_tags'][i] = self._read_words(filename)
            features['syntax'][i] = self._read_syntax(filename)
        return features


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data[self.key]


class ProbabilityThreshold(BaseEstimator, ClassifierMixin):
    """ Sets uncertain predictions to <UNK> """

    def __init__(self, estimator, pt):
        self.estimator = estimator
        self.pt = pt

    def fit(self, X, y):
        self.classes_ = list(set(y))
        self.estimator.fit(X, y)

    def predict(self, X):
        proba = self.estimator.predict_proba(X)
        y = self.estimator.classes_[np.argmax(proba, axis=1)]

        proba.sort(axis=1)
        diff = proba[:, -1] - proba[:, -2]

        y[diff < self.pt] = '<UNK>'
        return y


def identity(x):
    return x


def _document_group_frequency(X, y=None):
    X = X.toarray()
    X[X > 0] = 1

    if y is None:
        return X.shape[0], X.sum(axis=0)

    group_df = pd.DataFrame(X).groupby(y).sum()

    # in_df = np.vstack([group_df.loc[label] for label in y])
    df = np.vstack([np.asarray(X.sum(axis=0)).flatten() - group_df.loc[label].values for label in y]) + 1

    n_samples = np.vstack([np.sum(y != label) for label in y]) + 1
    return n_samples, df

# class GroupTfidfTransformer(TfidfTransformer):

#     def fit(self, X, y=None):
#         assert y is not None

#         super(GroupTfidfTransformer, self).fit(X, y)

#         dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

#         n_samples, n_features = X.shape
#         n_samples, df = self._document_frequency(X, dtype=dtype)
#         n_group_samples, group_df = self._document_frequency(X, y, dtype=dtype)

#         # perform idf smoothing if required
#         df += int(self.smooth_idf)
#         group_df += int(self.smooth_idf)
#         n_samples += int(self.smooth_idf)
#         n_group_samples += int(self.smooth_idf)

#         # log+1 instead of log makes sure terms with zero idf don't get
#         # suppressed entirely.

#         idf = np.log(n_samples / df) + 1
#         self._idf_diag = sp.sparse.diags(idf, offsets=0,
#                                          shape=(n_features, n_features),
#                                          format='csr',
#                                          dtype=dtype)

#         group_idf = np.log(n_group_samples / group_df) + 1
#         self._group_idf = group_idf

#         return self

#     def transform(self, X, y=None, copy=True):
#         if y is not None:
#             if sp.sparse.issparse(X):
#                 X = X.toarray()
#             return X * self._group_idf

#         return super(GroupTfidfTransformer, self).transform(X, copy=copy)


class GroupTfidfVectorizer(TfidfVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False,
                 min_group_df=1):

        super(GroupTfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf)

        self.min_group_df = min_group_df

    #     if use_gidf:
    #         self._tfidf = GroupTfidfTransformer(norm=norm, use_idf=use_idf,
    #                                             smooth_idf=smooth_idf,
    #                                             sublinear_tf=sublinear_tf)
    #     else:
    #         self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
    #                                        smooth_idf=smooth_idf,
    #                                        sublinear_tf=sublinear_tf)

    def fit_transform(self, raw_documents, y=None):
        self._check_params()
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        X = self._tfidf.transform(X, copy=False)
        return self._limit_group_features(X, y)

    def _limit_group_features(self, X, y):
        n_samples, y_df = _document_group_frequency(X, y)

        if sp.sparse.issparse(X):
            X = X.tolil()

        if self.min_group_df > 1:
            X[y_df < self.min_group_df] = 0

        return X


class SyntacticGroupTfidfVectorizer(GroupTfidfVectorizer):
    """ Uses ngrams based on dependency syntax hierarchy """

    def _dep_ngrams(self, dep_df):
        """ analyzer can be one of [word, lemma, dep] """

        min_n, max_n = self.ngram_range
        ngrams = []

        unigrams = []

        sent_ids = dep_df['dep'].index.get_level_values(0).unique()
        for sent_id in sent_ids:
            sent_df = dep_df.loc[sent_id]

            tokens = sent_df[self.analyzer]
            heads = sent_df['g']
            for idx, head in heads.items():
                ngram = [tokens[idx]]
                unigrams.append((ngram, idx, head))

                if min_n == 1:
                    ngrams.append(tuple(ngram))

                while len(ngram) < max_n and head > 0 and head in tokens:
                    ngram.append(tokens[head])
                    head = heads[head]

                    if len(ngram) >= min_n:
                        ngrams.append(tuple(ngram))

        if len(ngrams) == 0:
            raise ValueError(str(sent_ids) + str(unigrams) + str(dep_df))

        return ngrams

    def build_analyzer(self):
        return self._dep_ngrams


class StratifiedSplitter():
    """ Split data based on task specifications """

    def __init__(self, n_problems=20, n_candidates=9, n_train_docs=7, n_test_docs=100, unk_ratio=0.5,
                 random_state=None):
        assert n_problems % 4 == 0

        self.n_problems = n_problems
        self.n_candidates = n_candidates
        self.n_train_docs = n_train_docs
        self.n_test_docs = n_test_docs
        self.unk_ratio = unk_ratio
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        labels, languages = y, groups
        rnd = np.random.RandomState(self.random_state)

        lang_set = sorted(set(languages))
        for lang in lang_set:
            lang_index = np.argwhere(languages == lang)[:, 0]

            for _ in range(self.n_problems // 4):
                candidates = labels[lang_index]
                candidate_set = list(set(candidates) - {'<UNK>', 'problem00020_candidate00000'})
                candidate_subset = rnd.choice(candidate_set, self.n_candidates, replace=False)

                train_index, test_index = [], []
                for candidate in candidate_subset:
                    candidate_index = lang_index[np.argwhere(candidates == candidate)[:, 0]]
                    rnd.shuffle(candidate_index)

                    train_index.extend(candidate_index[:self.n_train_docs])
                    test_index.extend(candidate_index[self.n_train_docs:self.n_train_docs + self.n_test_docs])

                unk_index = lang_index[np.argwhere(np.isin(labels[lang_index], candidate_subset, invert=True))[:, 0]]
                rnd.shuffle(unk_index)
                test_index.extend(unk_index[:int(len(test_index) * self.unk_ratio)])

                rnd.shuffle(train_index)
                rnd.shuffle(test_index)
                yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_problems


def merge_probleminfo(collectioninfo, path):
    filenames, labels, languages = [], [], []

    for probleminfo in collectioninfo:
        lang = probleminfo['language']
        prob_name = probleminfo['problem-name']

        prob_filenames, prob_labels = get_model_inputs(probleminfo['candidates'])
        prob_filenames.extend(probleminfo['unknown'])
        prob_labels.extend(get_test_labels(probleminfo['unknown'], collectioninfo, path))

        filenames.extend(prob_filenames)
        labels.extend([prob_name + '_' + f if f != '<UNK>' else '<UNK>' for f in prob_labels])
        languages.extend([lang] * len(prob_filenames))

    return np.array(filenames), np.array(labels), np.array(languages)


def f1_macro(estimator, X, y_true):
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    y_pred = estimator.predict(X)
    return f1_score(y_true, y_pred,
                    labels=estimator.classes_,
                    average='macro')


class VisualTransformer(BaseEstimator, TransformerMixin):  # TODO Fix ngram_range (2,2) and (3,3)
    def __init__(self, ngram_range=(1, 1)):
        self.vocab = None
        self.ngram_range = ngram_range

    def _ngrams(self, x):
        punct_set = ['.', ',', ':', ';', '-', '?', '!', '(', ')', '\'', '"']
        x_puncts = [t for t in x if t in punct_set]

        min_n, max_n = self.ngram_range

        ngrams = []
        for n in range(min_n, max_n + 1):
            ngrams.extend([tuple(x_puncts[i:i+n]) for i in range(len(x_puncts) - n + 1)])

        return ngrams

    def fit(self, X, y=None):
        ngrams = np.unique([ngram for x in X for ngram in self._ngrams(x)])
        self.vocab = list(ngrams)
        return self

    def transform(self, X):
        X_ngrams = [self._ngrams(x) for x in X]
        return [[np.sum([n == ngram for n in x]) for ngram in self.vocab] for x in X_ngrams]


pipeline = Pipeline([
    ('raw_features', FeatureExtractor()),
    ('feature_union', FeatureUnion(
        [
            # Select best character n-gram features  char_ngrams__text__ngram_range
            ('visual', Pipeline([
                ('text', FeatureSelector('tokens')),
                ('features', VisualTransformer(ngram_range=(1, 2))),
                ('scaler', MaxAbsScaler())
            ])),

            # Select best character n-gram features  char_ngrams__text__ngram_range
            ('char_ngrams', Pipeline([
                ('text', FeatureSelector('text')),
                ('tfidf', GroupTfidfVectorizer(analyzer='char', lowercase=False,
                                               ngram_range=(2, 4), use_idf=True, min_group_df=1)),
                ('scaler', MaxAbsScaler()),
                ('best', TruncatedSVD(n_components=150))
            ])),

            # Select best token n-gram features
            ('token_ngrams', Pipeline([
                ('text', FeatureSelector('tokens')),
                ('tfidf', GroupTfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=5,
                                               lowercase=False, tokenizer=identity, use_idf=False)),
                ('scaler', MaxAbsScaler()),
                ('best', TruncatedSVD(n_components=150))
            ])),

            # Select best POS tag n-gram features
            ('pos_ngrams', Pipeline([
                ('text', FeatureSelector('pos_tags')),
                ('tfidf', GroupTfidfVectorizer(analyzer='word', tokenizer=identity, lowercase=False,
                                               ngram_range=(1, 4), use_idf=False, min_group_df=1)),
                ('scaler', MaxAbsScaler()),
                ('best', TruncatedSVD(n_components=150))
            ])),

            # Select best dependency tag sn-gram features
            ('dep_sngrams', Pipeline([
                ('text', FeatureSelector('syntax')),
                ('tfidf', SyntacticGroupTfidfVectorizer(analyzer='dep',
                                                        ngram_range=(1, 2), use_idf=False, min_group_df=3)),
                ('scaler', MaxAbsScaler()),
                ('best', TruncatedSVD(n_components=150))
            ])),

            # Select best word sn-gram features
            ('token_sngrams', Pipeline([
                ('text', FeatureSelector('syntax')),
                ('tfidf', SyntacticGroupTfidfVectorizer(analyzer='word',
                                                        ngram_range=(2, 3), use_idf=False, min_group_df=3)),
                ('scaler', MaxAbsScaler()),
                ('best', TruncatedSVD(n_components=150))
            ]))
        ],
    )),
    ('svc', ProbabilityThreshold(
        CalibratedClassifierCV(LinearSVC(), cv=5),
        pt=0.1)
     ),
])


def predict_all_problems(path, outpath, features, evaluate=False):
    start_time = time.time()

    lang_scores = {'en': [], 'fr': [], 'it': [], 'sp': []}

    collectioninfo = read_collection_info(path)
    for probleminfo in collectioninfo:
        print('Solving problem {} [{}]'.format(probleminfo['problem-name'], probleminfo['language']))

        train_filenames, train_labels = get_model_inputs(probleminfo['candidates'])
        test_filenames = probleminfo['unknown']

        pipeline.set_params(
            feature_union__transformer_weights={
                'visual': 1 if 'visual' in features else 0,
                'char_ngrams': 1 if 'char_ngrams' in features else 0,
                'token_ngrams': 1 if 'token_ngrams' in features else 0,
                'pos_ngrams': 1 if 'pos_ngrams' in features else 0,
                'dep_sngrams': 1 if 'dep_sngrams' in features else 0,
                'token_sngrams': 1 if 'token_sngrams' in features else 0
            }
        )

        pipeline.fit(train_filenames, train_labels)
        predictions = list(pipeline.predict(test_filenames))

        if evaluate:
            test_labels = get_test_labels(test_filenames, collectioninfo, path)
            prob_scores = precision_recall_fscore_support(test_labels, predictions,
                                                          labels=list(probleminfo['candidates'].keys()),
                                                          average='macro')
            lang_scores[probleminfo['language']].append(prob_scores)
            print(*prob_scores)

        save_output(outpath, probleminfo['problem-name'], test_filenames, predictions)

    if evaluate:
        p, r, f = [], [], []
        for lang, scores in lang_scores.items():
            lp, lr, lf = [], [], []
            for prob_scores in scores:
                lp.append(prob_scores[0])
                lr.append(prob_scores[1])
                lf.append(prob_scores[2])
            print(lang + ':', np.mean(lp), np.mean(lr), np.mean(lf))
            p.append(lp)
            r.append(lr)
            f.append(lf)
        print('Overall:', np.mean(p), np.mean(r), np.mean(f))

    print('elapsed time:', time.time() - start_time)


def grid_search(path, outpath, features, evaluate=False):
    collectioninfo = read_collection_info(path)
    filenames, labels, languages = merge_probleminfo(collectioninfo, path)

    n_problems = 40

    param_grids = []

    param_grids.append({
        'feature_union__visual__features__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)]
    })

    for f in ['char_ngrams', 'pos_ngrams', 'dep_sngrams', 'token_sngrams']:
        ngram_range = [(i, j) for i in [1, 2] for j in [1, 2, 3, 4] if j >= i]
        param_grids.append({
            'feature_union__{}__tfidf__ngram_range'.format(f):  ngram_range,
            'feature_union__{}__tfidf__use_idf'.format(f):      [True, False],
            'feature_union__{}__tfidf__min_group_df'.format(f): range(1, 5),
        })

    param_grids.append({
        'svc__pt': [0.01, 0.05, 0.1, 0.15, 0.2]
    })

    for i, param_grid in enumerate(param_grids):
        print('Starting grid search {}'.format(i))

        cv = GridSearchCV(pipeline, param_grid, f1_macro, n_jobs=-1, iid=False,
                          cv=StratifiedSplitter(n_problems=n_problems, random_state=123),
                          error_score=np.nan, return_train_score=False, refit=False, verbose=3)
        cv.fit(X=filenames, y=labels, groups=languages)

        print('Search finished. Best mean macro f1-score is {} [{}]\n'.format(cv.best_score_, cv.best_index_))

        with open(outpath + os.sep + 'cv_results_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(cv.cv_results_, f)

        with open(outpath + os.sep + 'best_params_{}.json'.format(i), 'w') as f:
            f.write(json.dumps(cv.best_params_))

        pipeline.set_params(**cv.best_params_)


def preprocess_input(raw_path, input_path):
    print(os.system('./scripts/runAll.sh {} {}'.format(raw_path, input_path)))


def main():
    parser = argparse.ArgumentParser(description='SVM Authorship Attribution Method')
    parser.add_argument('-r', type=str, help='Path to the main folder of the preprocessed input data')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    parser.add_argument('-o', type=str, help='Path to an output folder')
    parser.add_argument('-f', type=str, default='visual,char_ngrams,token_ngrams,pos_ngrams,dep_sngrams,token_sngrams',
                        help='Features [visual,char_ngrams,token_ngrams,pos_ngrams,dep_sngrams,token_sngrams]')
    parser.add_argument('--eval', dest='eval', action='store_const',
                        const=True, default=False,
                        help='Evaluate results')
    parser.add_argument('--grid-search', dest='action', action='store_const',
                        const=grid_search, default=predict_all_problems,
                        help='Perform grid search')
    args = parser.parse_args()

    if not args.i:
        print('ERROR: The input folder is required')
        parser.exit(1)
    if not args.o:
        print('ERROR: The output folder is required')
        parser.exit(1)

    if args.r:
        preprocess_input(args.r, args.i)

    features = args.f.split(',')

    args.action(args.i, args.o, features, args.eval)


if __name__ == '__main__':
    main()
