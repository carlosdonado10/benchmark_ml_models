from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier


class Estimator:
    def __init__(self, name, model, multilabel=False):
        self.name = name
        if multilabel:
            self.model = OneVsRestClassifier(model)
        else:
            self.model = model

    def fit(self, dataset):
        self.model.fit(dataset.data, dataset.target)

    def predict(self, dataset):
        try:
            return self.model.predict_proba(dataset.data)
        except AttributeError as e:
            return self.model.decision_function(dataset.data)

    def auc(self, dataset):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_score = self.predict(dataset)
        for i in dataset.label_mapping.values():
            fpr[i], tpr[i], _ = roc_curve(dataset.target[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(dataset.target.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        return roc_auc['micro']
