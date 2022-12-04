from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticClassification
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import sklearn
import sys
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Clean up the reviews data
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


#
# def normalize_sentiment( sentiment ):
#     return int( sentiment == 'positive' )

def get_review():
    review = input("Please give a review: " )
    print(review)
    return review

def main():

    # imdb = pd.read_csv("IMDB Dataset.csv")
    # print("imdb length", len(imdb))
    #
    # imdb['sentiment'] = imdb['sentiment'].apply(normalize_sentiment)

    # All training data
    train = pd.read_csv("labeledTrainData.tsv", header=0, \
                        delimiter="\t", quoting=3)

    print("train length", len(train))
    #
    # frames = [train, imdb]
    #
    # train = pd.concat(frames)
    # print("new train length", len(train))

    # All test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                       quoting=3)

    print("test length", len(test))
    # print(data.head)
    # split data into train and test
    # train_data = train.sample(frac=0.7)
    # test = train.drop(train.index)

    #
    xs = train[['review']]   # Training reviews
    ys = train['sentiment']  # Traing target

    x_test = test[['review']]
    # print(xs)
    user_review = get_review()
    # d = {'review': [user_review]}
    # df = pd.DataFrame(data=d)
    # print(df[['review']])
    # x_test = df[['review']]
    x_test = test[['review']]
    # y_test = test['sentiment']

    # xs = data[['review']]
    # ys = data['sentiment']

    # print(xs)
    # print(ys)

    # test preprocessor outside of pipeline to see if it works
    # t = Preprocessor()
    # xs_lower = t.transform(xs)
    # print(xs_lower)

    # vectorize the train data for logistic regression
    # vectorizer = \
    #     CountVectorizer(
    #         max_df=0.1,
    #         min_df=2,
    #         strip_accents='unicode',
    #         max_features=100000,
    #         ngram_range=(1, 3),
    #     )

    # vectorizer for random forest
    vectorizer = \
        CountVectorizer(
            max_df=0.1,
            min_df=5,
            strip_accents='unicode',
            max_features=100000,
            ngram_range=(1, 3),
        )
    # steps
    # steps = [
    #     ('tokenize', Preprocessor()),
    #     ('vectorize', vectorizer),
    #     ('classify', LogisticRegression(solver='lbfgs', max_iter=1000))
    # ]

    steps = [
        ('tokenize', Preprocessor()),
        ('vectorize', vectorizer),
        ('classify', RandomForestClassifier())
    ]

    # Grid for vectorizor
    grid = {
        # 'vectorize__ngram_range': [(1, 1), (1, 3), (2, 3)],
        # 'vectorize__max_df': [1.0, 0.7, 0.1],
        # 'vectorize__min_df': [2, 5, 10, 100],
        # 'vectorize__max_features': [1000, 10000, 100000],
        'classify__max_depth': list(np.arange(10, 100, step=10)) + [None],
        "classify__n_estimators": np.arange(10, 500, step=50),
        'classify__max_features': randint(1, 7),
        'classify__criterion': ['gini', 'entropy'],
        'classify__min_samples_leaf': randint(1, 4),
        'classify__min_samples_split': np.arange(2, 10, step=2)
    }

    # Grid for random forest
    # rs_space = {'max_depth': list(np.arange(10, 100, step=10)) + [None],
    #             'n_estimators': np.arange(10, 500, step=50),
    #             'max_features': randint(1, 7),
    #             'criterion': ['gini', 'entropy'],
    #             'min_samples_leaf': randint(1, 4),
    #             'min_samples_split': np.arange(2, 10, step=2)
    #             }


    pipe = Pipeline(steps)
    pipe.fit(xs, ys)
    prediction = pipe.predict(x_test)

    print(prediction)

    # print(list(y_test))


    # score = metrics.accuracy_score(y_test, prediction)

    # print("accuracy is: ", score)

    # pass in test reviews to classify
    # print(pipe.predict_proba([['Thought this movie was very cool.'], ['Thought this movie was lame.']]))

    # Comment in for the
    search = RandomizedSearchCV(pipe, grid, scoring='accuracy', n_jobs=-1)
    search.fit(xs, ys)

    print(search.best_score_) # r-squared score
    print(search.best_params_) # hyperparameters


    # grid_forest = RandomizedSearchCV(pipe,rs_space,scoring='accuracy')
    # grid_forest.fit(xs, ys)
    #
    # print(grid_forest.best_score_)  # r-squared score
    # print(grid_forest.best_params_)  # hyperparameters

    output = pd.DataFrame(data={'id': test['id'], "sentiment": prediction})
    output.to_csv("Baf_of_Words_model.csv", index=False, quoting=3)
if __name__ == "__main__":
    main()
