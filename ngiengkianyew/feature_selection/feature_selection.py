from sklearn import datasets
import numpy as np
import pandas as pd
data = datasets.load_iris()
X = data.data
y = data.target
y = np.array([1 if yi >= 1 else yi for yi in y])
X = pd.DataFrame(X, columns=data.feature_names)