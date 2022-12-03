from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import sklearn
import sys
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Preprocessor(BaseEstimator, TransformerMixin):

    def fit(self, xs, ys=None):
        return self

    def transform(self, xs):
        def de_tag(t):
            return re.sub('<br/>', '', t)

        def drop_quote_and_hyphen(t):
            return re.sub(r"\'|\-", '', t)

        def spacify_non_letter_or_digit(t):
            return re.sub('\W', ' ', t)

        def combine_spaces(t):
            return re.sub('\s+', ' ', t)

        transformed = xs['review'].str.lower()
        transformed = transformed.apply(de_tag)
        transformed = transformed.apply(drop_quote_and_hyphen)
        transformed = transformed.apply(spacify_non_letter_or_digit)
        transformed = transformed.apply(combine_spaces)

        return transformed


def main():
    data = pd.read_table('./labeledTrainData.tsv')
    # print(data)
    # print(type(data))
    # print("# row", len(data))
    # print("# col", len(data.columns))

    # print(data.head)
    # split data into train and test
    train = data.sample(frac=0.7)
    test = data.drop(train.index)

    xs = train[['review']]
    ys = train['sentiment']

    x_test = test[['review']]
    y_test = test['sentiment']

    # xs = data[['review']]
    # ys = data['sentiment']

    # print(xs)
    # print(ys)

    # test preprocessor outside of pipeline to see if it works
    # t = Preprocessor()
    # xs_lower = t.transform(xs)
    # print(xs_lower)

    # vectorize the train data
    vectorizer = \
        CountVectorizer(
            max_df=0.9,
            min_df=5,
            strip_accents='unicode',
            max_features=100000,
            ngram_range=(1, 3),
        )

    steps = [
        ('tokenize', Preprocessor()),
        ('vectorize', vectorizer),
        ('classify', LogisticRegression(solver='lbfgs', max_iter=1000))
    ]
    grid = {
        'vectorize__ngram_range': [(1, 1), (1, 3), (2, 3)],
        'vectorize__max_df': [1.0, 0.7, 0.1],
        'vectorize__min_df': [2, 5, 10, 100],
        'vectorize__max_features': [1000, 10000, 100000],
        # ‘normalize__norm’: [ ‘l1’, ‘l2’, ‘max’ ],
    }
    pipe = Pipeline(steps)
    pipe.fit(xs, ys)
    prediction = pipe.predict(x_test)
    print(prediction)
    print(y_test)
    score = metrics.accuracy_score(y_test, prediction)
    print("accuracy is: ", score)

    # pass in test reviews to classify
    # print(pipe.predict_proba(['Thought this movie was very cool.', 'Thought this movie was lame.']))

    # search = RandomizedSearchCV(pipe, grid, scoring='r2', n_jobs = -1)
    # search = RandomizedSearchCV(pipe, grid, scoring='r2', n_jobs=-1)
    # search.fit(xs, ys)
    #
    # print(search.best_score_) # r-squared score
    # print(search.best_params_) # hyperparameters


if __name__ == "__main__":
    main()
