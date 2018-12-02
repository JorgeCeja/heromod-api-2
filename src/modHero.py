import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


class ModHero(object):
    def __init__(self, models_path, max_feat, labels):
        self.models_path = models_path
        self.max_feat = max_feat
        self.labels = labels
        self.models = self.load_models(self.models_path)
        self.vect = TfidfVectorizer(
            max_features=self.max_feat, stop_words='english')

    def load_models(self, models_path):
        models = []

        for label in self.labels:
            print(label)
            clf = joblib.load(self.models_path+label+".pickle")
            models.append(clf)

        return models

    def classify(self, _input):
        # preprocess text
        input_dmt = self.vect.fit_transform([_input])

        results = {}
        for idx, label in enumerate(self.labels):
            results[label] = self.models[idx].predict(input_dmt)

        return results
