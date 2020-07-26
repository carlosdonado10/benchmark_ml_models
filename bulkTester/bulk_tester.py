from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC,NuSVC
from Predictor.predictor import Estimator
multilabel_classification_estimators = [
    Estimator('KNN', KNeighborsClassifier(), multilabel=True),
    Estimator('LDA', LinearDiscriminantAnalysis(), multilabel=True),
    Estimator('QDA', QuadraticDiscriminantAnalysis(), multilabel=True),
    Estimator('Random Forest', RandomForestClassifier(), multilabel=True),
    Estimator('AdaBoost', AdaBoostClassifier(), multilabel=True),
    Estimator('GradientBoosting', GradientBoostingClassifier(), multilabel=True),
    Estimator('DecisionTree', DecisionTreeClassifier(), multilabel=True),
    Estimator('Extra Tree ', ExtraTreeClassifier(), multilabel=True),
    Estimator('Gaussian Naive Bayes', GaussianNB(), multilabel=True),
    Estimator('Multinomial Naive Bayes', MultinomialNB(), multilabel=True),
    Estimator('Support Vector Classifier', SVC(), multilabel=True),
    Estimator('Nu-Support Vector Classifier', NuSVC(), multilabel=True),
    # Estimator('Linear SGD', SGDClassifier(), multilabel=True)
]


class ProblemTypes:
    REGRESSION = 0
    CLASSIFICATION = 1
    MULTILABEL_CLASSIFICATION = 2


class BulkTester:
    def __init__(self, dataset, estimators=None, problem_type=None):
        if estimators:
            self.estimators = estimators
        elif problem_type == ProblemTypes.MULTILABEL_CLASSIFICATION:
            self.estimators = multilabel_classification_estimators

        self.ds = dataset

    def benchmark(self):
        auc_list = []
        for estimator in self.estimators:
            estimator.fit(self.ds)
            auc = estimator.auc(self.ds)
            print(f'estimator: {estimator.name} yields AUC: {auc}')
            auc_list.append(auc)
        return zip(self.estimators, auc_list)
