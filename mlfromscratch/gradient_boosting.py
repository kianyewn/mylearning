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
    
    
def divide_on_feature(self, X, feature_i, threshold):
    if isinstance(threshold, int) or isinstance(threshold, float):
        filter_cond = lambda x: x[feature_i] >= threshold
    else:
        filter_con = lambda x: x[feature_i] == threshold
        
    # X1 = X[[filter_cond(sample) for sample in X]]
    # X2 = X[[not filter_cond(sample) for sample in X]]
    X_1 = np.array([sample for sample in X if filter_cond(sample)])
    X_2 = np.array([sample for sample in X if not filter_cond(sample)])
    return np.array([X_1, X_2])

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
                 min_samples,
                 max_depth,
                 min_impurity):
        self.min_samples = min_samples    
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        
    def _build_tree(self, X, y, current_depth):
        if len(y.shape) == 1:
            y = y.rehape(-1,1)
        
        n_samples, n_features = X.shape
        Xy = np.concatenate([X, y], axis=1)
        largest_impurity = 0
        best_criteria = None
        best_splits = None
        if n_samples >= self.min_samples and current_depth <= self.max_depth: 
            for feature_i in range(len(n_features)):
                unique_values = np.unique(X[:, feature_i])
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy,feature_i, threshold)
                    
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
                
        
# class GradientBoosting(object):
#     """Super class of GradientBoostingClassifier and GradientBoostinRegressor. 
#     Uses a collection of regression trees that trains on predicting the gradient
#     of the loss function. 

#     Parameters:
#     -----------
#     n_estimators: int
#         The number of classification trees that are used.
#     learning_rate: float
#         The step length that will be taken when following the negative gradient during
#         training.
#     min_samples_split: int
#         The minimum number of samples needed to make a split when building a tree.
#     min_impurity: float
#         The minimum impurity required to split the tree further. 
#     max_depth: int
#         The maximum depth of a tree.
#     regression: boolean
#         True or false depending on if we're doing regression or classification.
#     """
#     def __init__(self, n_estimators, learning_rate, min_samples_split,
#                  min_impurity, max_depth, regression):
#         self.n_estimators = n_estimators
#         self.learning_rate = learning_rate
#         self.min_samples_split = min_samples_split
#         self.min_impurity = min_impurity
#         self.max_depth = max_depth
#         self.regression = regression
#         self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        
#         # Square loss for regression
#         # Log loss for classification
#         self.loss = SquareLoss()
#         if not self.regression:
#             self.loss = CrossEntropy()

#         # Initialize regression trees
#         self.trees = []
#         for _ in range(n_estimators):
#             tree = RegressionTree(
#                     min_samples_split=self.min_samples_split,
#                     min_impurity=min_impurity,
#                     max_depth=self.max_depth)
#             self.trees.append(tree)


#     def fit(self, X, y):
#         y_pred = np.full(np.shape(y), np.mean(y, axis=0))
#         for i in self.bar(range(self.n_estimators)):
#             # predict the residual. The residual is the gradient - (y_true - y_pred) from MSEloss
#             gradient = self.loss.gradient(y, y_pred)
#             # fit on the residual
#             self.trees[i].fit(X, gradient)
#             # predict the residual. When we make a prediction, we are calculating the average of the observed yi, this is the gamma
#             update = self.trees[i].predict(X)
#             # Update y prediction by adding the residual
#             y_pred -= np.multiply(self.learning_rate, update)


#     def predict(self, X):
#         y_pred = np.array([])
#         # Make predictions
#         for tree in self.trees:
#             update = tree.predict(X)
#             update = np.multiply(self.learning_rate, update)
#             y_pred = -update if not y_pred.any() else y_pred - update

#         if not self.regression:
#             # Turn into probability distribution
#             y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
#             # Set label to the value that maximizes probability
#             y_pred = np.argmax(y_pred, axis=1)
#         return y_pred


# class GradientBoostingRegressor(GradientBoosting):
#     def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
#                  min_var_red=1e-7, max_depth=4, debug=False):
#         super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators, 
#             learning_rate=learning_rate, 
#             min_samples_split=min_samples_split, 
#             min_impurity=min_var_red,
#             max_depth=max_depth,
#             regression=True)

# class GradientBoostingClassifier(GradientBoosting):
#     def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
#                  min_info_gain=1e-7, max_depth=2, debug=False):
#         super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators, 
#             learning_rate=learning_rate, 
#             min_samples_split=min_samples_split, 
#             min_impurity=min_info_gain,
#             max_depth=max_depth,
#             regression=False)

#     def fit(self, X, y):
#         y = to_categorical(y)
#         super(GradientBoostingClassifier, self).fit(X, y)