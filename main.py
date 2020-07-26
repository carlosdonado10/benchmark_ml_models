import pandas as pd
from DataSet.dataset import Dataset
from Predictor.predictor import Estimator
from sklearn.neighbors import KNeighborsClassifier
from bulkTester.bulk_tester import BulkTester, ProblemTypes
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.columns = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)', 'Class']

# ds = Dataset(df, label_encoder='MultiLabel', label='Class')
#
# bulk = BulkTester(ds, problem_type=ProblemTypes.MULTILABEL_CLASSIFICATION)
# bulk.benchmark()
