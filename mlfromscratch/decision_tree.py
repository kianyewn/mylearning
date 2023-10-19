from sklearn import datasets
import pandas as pd
import math
import numpy as np

def calculate_entropy(y: pd.Series):
    log2 = lambda x: math.log(x) / math.log(2)
    probs = y.value_counts(normalize=True)
    entropy =  -(probs  * probs.apply(lambda x: log2(x))).sum()
    return entropy

def calculate_entropy(y: np.array):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_values = np.unique(y)
    total_cnt = y.shape[0]
    entropies = []
    for y_i in unique_values:
        cnt = (y == y_i).sum()
        prob = cnt / total_cnt
        entropies.append(-prob * log2(prob))
    return sum(entropies)

# test entropy
labels = [0,0,0,0,0,0,0,1,1,1]
y = pd.Series(labels)
calculate_entropy(y)
    
import scipy.stats as stats

probabilities = [0,3,0,7]
calculated_entropy = stats.entropy(probabilities, base=2)
calculated_entropy 

def divide_on_feature(Xy, feature_i, threshold):
    if isinstance(threshold, int) or isinstance(threshold, float):
        filter_cond = lambda sample: sample[feature_i] == threshold
    elif isinstance(threshold, str):
        filter_cond = lambda sample: sample[feature_i] >= threshold
    X_1 = np.array([sample for sample in Xy if filter_cond(sample)])
    X_2 = np.array([sample for sample in Xy if not filter_cond(sample)])
    return np.array([X_1, X_2])

dataset = datasets.load_iris()
X, y = dataset.data, dataset.target
a, b = divide_on_feature(X, 0, threshold=5.0)

def divide_on_feature(X, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])
a2, b2 = divide_on_feature(X, 0, threshold=5.0)
# assert np.allclose(a, a2) == True
# assert np.allclose(b, b2) == True

def calculate_entropy_split(y, y1, y2):
    # entropy split
    y_n_samples = y.shape[0]
    left_n_samples = y1.shape[0]
    right_n_samples = y2.shape[0]
    entropy_left = calculate_entropy(y1)
    entropy_right = calculate_entropy(y2)
    
    entropy_split = (left_n_samples/y_n_samples) * entropy_left +\
        (right_n_samples/y_n_samples) * entropy_right
    return entropy_split

def calculate_information_gain(y, y1, y2):
    entropy = calculate_entropy(y)
    entropy_split = calculate_entropy_split(y, y1, y2)
    return entropy - entropy_split

def _calculate_information_gain(y, y1, y2):
    # Calculate information gain
    p = len(y1) / len(y)
    entropy = calculate_entropy(y)
    info_gain = entropy - p * \
        calculate_entropy(y1) - (1 - p) * \
        calculate_entropy(y2)

    return info_gain

y = np.array([1,1,1,1, 0,0,0,0,0,1])
y1 = np.array([1,1,1,1])
y2 = np.array([0,0,0,0,0,1])
calculate_entropy_split(y, y1, y2) # == 0.390, https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
calculate_information_gain(y, y1, y2) # ~= 0.61
# _calculate_information_gain(y, y1, y2)

def majority_voting(y):
    unique_values = np.unique(y)
    current_max_vote = 0
    best_vote = None
    for yi in unique_values:
        cnt = (y==yi).sum()
        if cnt > current_max_vote:
            current_max_vote = cnt
            best_vote = yi
    return best_vote

y = [1,1,1,0,0]
assert majority_voting(y) == 1
y= [1,0,0,0,1]
assert majority_voting(y) == 0


from dataclasses import dataclass
from typing import Any
@dataclass
class DecisionNode:
    feature_i: Any = None
    threshold: Any = None
    value: Any = None
    true_branch: Any = None
    false_branch: Any = None
        
class DecisionTree:
    def __init__(self,
                 min_samples_split=2,
                 min_impurity=1e-7,
                 max_depth =float('inf'),
                 loss= None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _build_tree(self, X, y, current_depth=0):
        
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        
        Xy = np.concatenate([X,y], axis=1)
        
        n_samples, n_features = np.shape(X)
        
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        
                        # calculate impurity
                        impurity = calculate_information_gain(y, y1, y2)
                        
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {'feature_i': feature_i,
                                             'threshold': threshold}
                            best_sets = {
                                'leftX': Xy1[:,:n_features],
                                'lefty': Xy1[:, n_features:],
                                'rightX': Xy2[:,:n_features],
                                'righty': Xy2[:,n_features:]
                            }
        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets['leftX'], best_sets['lefty'], current_depth = current_depth+1)
            false_branch = self._build_tree(best_sets['rightX'], best_sets['righty'], current_depth = current_depth+1)
            return DecisionNode(feature_i=best_criteria['feature_i'], 
                                threshold=best_criteria['threshold'],
                                true_branch=true_branch,
                                false_branch=false_branch)
       
       # if the impurity is lower than min_impurity, return as a leaf node 
        leaf_value = majority_voting(y)
        return DecisionNode(value=leaf_value)
    
    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]
        
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
                
        elif feature_value == tree.threshold:
            branch = tree.true_branch
            
        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(x) for x in X]
        return y_pred
    
    
dataset = datasets.load_iris()
X, y = dataset.data, dataset.target
y = np.array([1 if yi > 1 else yi for yi in y])
clf = DecisionTree()
clf.fit(X, y)

preds = clf.predict(X)
preds
