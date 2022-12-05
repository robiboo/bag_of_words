from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import re
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

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


def get_review():
    review = input("Please give a movie review. Tell us what you thought of the plot, acting, cinematography, etc. The more the better! " )
    # print(review)
    return review

def normalize_sentiment(sentiment):
    return int(sentiment == 'positive')
def demo(steps):
    # IMDB dataset for the demo
    train = pd.read_csv("IMDB Dataset.csv", header=0)
    train['sentiment'] = train['sentiment'].apply(normalize_sentiment)
    xs = train[['review']]  # Training reviews
    ys = train['sentiment']  # Traing target
    user_review = get_review()
    d = {'review': [user_review]}
    df = pd.DataFrame(data=d)
    x_test = df[['review']]
    pipe = Pipeline(steps)
    pipe.fit(xs, ys)
    prediction = pipe.predict(x_test)
    if prediction[0] == 0:
        print("You hated this movie!")
    else:
        print("you loved this movie <3")
      
def train_test(data, steps):
    # split data into train and test
    train = data.sample(frac=0.7)
    test = data.drop(train.index)
   
    xs = train[['review']]  # Training reviews
    ys = train['sentiment']  # Training target

    x_test = test[['review']]
   
    y_test = test['sentiment']
    pipe = Pipeline(steps)
    pipe.fit(xs, ys)
    prediction = pipe.predict(x_test)
    score = metrics.accuracy_score(y_test, prediction)

    print("accuracy is: ", score)

def kaggle_prediction(train, test, steps):
    xs = train[['review']]  # Training reviews
    ys = train['sentiment']  # Traing target

    x_test = test[['review']]
    pipe = Pipeline(steps)
    pipe.fit(xs, ys)
    prediction = pipe.predict(x_test)

    output = pd.DataFrame(data={'id': test['id'], "sentiment": prediction})
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)


def logistic_regression_grid(train, steps):
    xs = train[['review']]  # Training reviews
    ys = train['sentiment']  # Traing target
    pipe = Pipeline(steps)
    grid_logistic = {
        'classify__C' : [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2],
    }
    search = RandomizedSearchCV(pipe, grid_logistic, scoring='accuracy', n_jobs=-1)
    search.fit(xs, ys)

    print(search.best_score_) # r-squared score
    print(search.best_params_) # hyperparameters

def random_forest_grid(train, steps):
    xs = train[['review']]  # Training reviews
    ys = train['sentiment']  # Traing target
    pipe = Pipeline(steps)
    grid_regression = {
        'classify__n_estimators' : list(range(10,101,10)),
        'classify__max_features' : list(range(6,32,5))
    }
    search = RandomizedSearchCV(pipe, grid_regression, scoring='accuracy', n_jobs=-1)
    search.fit(xs, ys)

    print(search.best_score_) # r-squared score
    print(search.best_params_) # hyperparameters

def main():
    # receive user input for which estimator to run
    estimator = input("Please put in \'l\' for logistic regression or \'r\' for random forest (default is l): ")
    # All training data
    train = pd.read_csv("labeledTrainData.tsv", header=0, \
                        delimiter="\t", quoting=3)

    # All test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                       quoting=3)

    if (estimator != 'l' and estimator != 'r'):
        print("Sorry you have given the wrong estimator!!")
        return
    
    # receive user input for which dataset or optimization to run
    mode = input("Which type of experiment would you like to do? Options are \ndemo (d), \nkaggle (k), \ntrain test (t), \
    \nOptimization (o): ")

    # vectorize the train data for logistic regression
    vectorizer_l = \
        CountVectorizer(
            max_df=0.1,
            min_df=2,
            strip_accents='unicode',
            max_features=100000,
            ngram_range=(1, 3),
        )

    # vectorizer for random forest
    vectorizer_r = \
        CountVectorizer(
            max_df=0.1,
            min_df=5,
            strip_accents='unicode',
            max_features=100000,
            ngram_range=(1, 3),
        )

    vectorizer = vectorizer_l
  
    if estimator == 'r':
        vectorizer = vectorizer_r

    steps_l = [
        ('tokenize', Preprocessor()),
        ('vectorize', vectorizer),
        ('normalize', Normalizer( norm = 'l1' ) ),
        ('classify', LogisticRegression(solver='lbfgs', max_iter=1000, C=100))
    ]

    steps_r = [
        ('tokenize', Preprocessor()),
        ('vectorize', vectorizer),
        ('normalize', Normalizer( norm = 'l1' ) ),
        ('classify', RandomForestClassifier(n_estimators=100, max_features=26))
    ]
    steps = steps_l
    if ( estimator == 'r'):
        steps = steps_r
    
    if (mode == 'd'):
        print("Demo version")
        demo(steps)
    elif (mode == 'k'):
        kaggle_prediction(train, test, steps)
        print('Submit Bag_of_Words_model.csv to Kaggle to get accuracy score')
    elif (mode == 't'):
        print('Sample train test split')
        train_test(train, steps)
    elif (mode == 'o' and estimator != 'r'):
        print('Logistic Regression optimization with Random Search CV')
        logistic_regression_grid(train, steps)
    elif (mode == 'o' and estimator == 'r'):
        print('Random Forest Classifier optimization with Random Search CV')
        random_forest_grid(train , steps)
    else:
        print('Sorry you have given the wrong mode!!')
    return
  

if __name__ == "__main__":
    main()
