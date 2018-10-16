from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def multi_nb(train, test):
    """Run naive bayes baseline"""
    pipeline = Pipeline(
        [("vectoriser", TfidfVectorizer()), ("classifier", MultinomialNB())]
    )

    train_x, train_y = train
    test_x, test_y = test

    pipeline.fit(train_x, train_y)
    return pipeline.score(test_x, test_y) * 100
