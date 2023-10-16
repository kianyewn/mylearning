import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class DecisionNode:
    feature_i: str = None 
    threshold:float = None
    left_branch: Any = None
    right_branch: Any = None
    value: float = None
    
def divide_on_feature(X, feature_i, threshold):
    if isinstance(threshold, int) or isinstance(threshold, float):
        filter_cond = lambda x: x[feature_i] >= threshold
    else:
        filter_cond = lambda x: x[feature_i] == threshold
        
    # X1 = X[[filter_cond(sample) for sample in X]]
    # X2 = X[[not filter_cond(sample) for sample in X]]
    X_1 = np.array([sample for sample in X if filter_cond(sample)])
    X_2 = np.array([sample for sample in X if not filter_cond(sample)])
    return [X_1, X_2]

def calculate_variance(X):
    """Calculate Variance based on 2-d np array (n_samples, features)
    """
    n_samples = len(X)
    mean = np.mean(X, axis=0) # (n_features,)
    variance =  (1/ n_samples) * np.diag((X - mean).T.dot((X-mean)))
    return variance
    
def variance_reduction(y, y1, y2):
    """Calculate the variance between y, y1, y2, each is a 2d tensor of shape (n_samples, 1)"""
    y_samples = len(y)
    y1_n_samples = len(y1)
    y2_n_samples = len(y2)
    
    y_var = calculate_variance(y)
    y1_var = calculate_variance(y1)
    y2_var = calculate_variance(y2)
    
    var_reduc = y_var - ((y1_n_samples / y_samples) * y1_var + (y2_n_samples/y_samples) * y2_var)
    return np.sum(var_reduc)
    
class RegressionDecisionTree:
    def __init__(self, 
                 min_samples=2,
                 max_depth=float('inf'),
                 min_impurity=1e-7):
        self.min_samples = min_samples    
        self.max_depth = max_depth
        self.min_impurity = min_impurity
       
        # to be build after calling self.fit 
        self.root = None
        
    def _build_tree(self, X, y, current_depth=0):
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        
        n_samples, n_features = X.shape
        Xy = np.concatenate([X, y], axis=1)
        largest_impurity = 0
        best_criteria = None
        best_splits = None
        if n_samples >= self.min_samples and current_depth <= self.max_depth: 
            for feature_i in range(n_features):
                unique_values = np.unique(X[:, feature_i])
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    
                    if len(Xy1) >0 and len(Xy2) > 0:
                        y1 = Xy2[:, n_features:]
                        y2 = Xy1[:, n_features:]
                        
                        impurity = variance_reduction(y, y1, y2)
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {'feature_i': feature_i,
                                             'threshold': threshold}
                            best_splits = {'left_branch_X': Xy1[:, :n_features],
                                           'left_branch_y': Xy1[:,n_features:],
                                           'right_branch_X': Xy2[:, :n_features],
                                           'right_branch_y': Xy2[:, n_features:]}
        if largest_impurity > self.min_impurity:
            left_branch = self._build_tree(X=best_splits['left_branch_X'],
                                            y= best_splits['left_branch_y'],
                                            current_depth = current_depth+1)
            right_branch = self._build_tree(X= best_splits['right_branch_X'],
                                            y = best_splits['right_branch_y'],
                                            current_depth= current_depth+1)
            
            return DecisionNode(feature_i = best_criteria['feature_i'],
                                threshold = best_criteria['threshold'],
                                left_branch = left_branch,
                                right_branch = right_branch)
        else:
            leaf_value = np.mean(y)
            return DecisionNode(value= leaf_value)
     
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self
   
    def predict_sample(self, xs, root=None):
        if root is None:
            root = self.root
        
        if root.value:
            return root.value
            
        feature_i = root.feature_i
        threshold = root.threshold
        
        feature_value = xs[feature_i]
        branch = root.right_branch
        if isinstance(threshold, float) or isinstance(threshold, int):
            if feature_value >= threshold:
                branch = root.left_branch
        if isinstance(threshold, str):
            if feature_value == threshold:
                branch = root.left_branch
        return self.predict_sample(xs, root=branch)
    def predict(self, X):
        pred = [self.predict_sample(xs) for xs in X]
        return pred
                
from sklearn import datasets
# dataset = datasets.load_iris()
dataset = datasets.load_diabetes()
X, y = dataset.data, dataset.target
# y = np.array([1 if yi > 1 else yi for yi in y])
clf2 = RegressionDecisionTree()           
# clf._build_tree(X, y)     
clf2.fit(X, y)
# clf.predict_sample(X[5:6])
pred2 = clf2.predict(X)
clf2.root
np.mean((y - pred2)**2)

# Gradient Boosting
class GradientBoosting:
    def __init__(self,
                 n_estimators=3,
                 max_depth=5, 
                 min_samples=2,
                 min_impurity=1e-7, 
                 learning_rate=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_impurity = min_impurity
        self.learning_rate = learning_rate
        
        self.trees = [RegressionDecisionTree(min_samples = self.min_samples,
                                             max_depth=self.max_depth,
                                             min_impurity=self.min_impurity) for n in range(self.n_esimators)]
        self.initial_pred = None
        
    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        y_pred = np.zeros(len(y))
        y_pred[:] = self.initial_pred
        for n in self.n_estimators:
            # gradient of mean squared error
            pseudo_residuals = - (y - y_pred)
            self.tree[n].fit(X, pseudo_residuals)
            # average residual is the prediction that will minimize the MSE loss function in this iteration
            pred = self.tree[n].predict(X)
            # negative for negative gradient descent
            y_pred = y_pred - self.learning_rate * pred
        return
    
    def predict(self, X):
        y_pred = np.zeros(len(X))
        y_pred[:] = self.initial_pred
        for tree in self.trees:
            pred = tree.predict(X)
            y_pred = y_pred - self.learning_rate * pred
        return y_pred
            
            
gbt = GradientBoosting()
gbt.fit(X,y)     

n_estimators = 3
max_depth = 10000
min_samples = 2
min_impurity = 1e-7
trees = [RegressionDecisionTree(max_depth=max_depth,
                                min_impurity=min_impurity,
                                min_samples=min_samples) for _ in range(n_estimators)]

trees[0].fit(X, y)
pred2 = trees[0].predict(X)
np.mean((y - pred2)**2)

initial_pred = y.mean()
y_preds = np.zeros((len(y)))
y_preds[:] = initial_pred
learning_rate = 0.1
for n in range(n_estimators):
    pseudo_residuals = y - y_preds
    trees[n].fit(X, pseudo_residuals)
    pred = np.array(trees[n].predict(X))
    # update model. Fm(x) = Fm-1(x) + alpha * new_residual prediction
    y_preds = y_preds + 0.5 * pred
    
## make predictions

y_pred = np.mean(y)
for tree in trees:
    pred = tree.predict(X)
    y_pred = y_pred + pred
    
    
    
# y_pred = np.array([])
y_pred = np.mean(y)
# Make predictions
for tree in trees:
    update = tree.predict(X)
    update = np.multiply(learning_rate, update)
    # y_pred = update if not y_pred.any() else y_pred + update
    y_pred = y_pred + update

loss2 = abs(y_pred - y).sum()
loss2 # 23981.051470588238


loss1 = abs(y_pred - y).sum()
loss1 # 67243.0



