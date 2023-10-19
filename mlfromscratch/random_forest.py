import numpy as np
import math


def calculate_entropy(y):
    unique_vals = np.unique(y)
    entropies = []
    base2 = lambda x: math.log(x) / math.log(2)
    for ys in unique_vals:
        prob = (y == ys).mean()
        entropies.append(prob * base2(prob))
    return - np.sum(entropies)

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
    return [X_1, X_2]
        
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

from dataclasses import dataclass
from typing import Any
@dataclass
class DecisionNode:
    feature_i: Any = None
    threshold: Any = None
    value: Any = None
    true_branch: Any = None
    false_branch: Any = None
    
class ClassificationDecisionTree:
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

        
from sklearn import datasets
dataset = datasets.load_iris()
X, y = dataset.data, dataset.target
y = np.array([1 if yi > 1 else yi for yi in y])
clf = ClassificationDecisionTree()
clf.fit(X, y)


def get_random_subsets(X, y, n_subsets, replacements=True):
    """ Return random subsets (with replacements) of the data """
    n_samples = np.shape(X)[0]
    # Concatenate x and y and do a random shuffle
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples      # 100% with replacements

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets

class RandomForestClassifier:
    def __init__(self, 
                 max_depth=5,
                 min_samples_split=2,
                 min_impurity=1e-7,
                 n_estimators= 3,
                 max_features=None):
        self.n_estimators = n_estimators
        self.trees = [ClassificationDecisionTree(min_samples_split=min_samples_split,
                                            max_depth=max_depth,
                                            min_impurity=min_impurity) for _ in range(self.n_estimators)]
        self.max_features = max_features
    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features is None:
            max_features = int(np.sqrt(n_features))
        
        # get one random subset of data for the trees
        subsets = get_random_subsets(X, y, self.n_estimators, replacements=True)

        for i in range(self.n_estimators):
            sub_X, sub_y = subsets[i]
            # print(sub_X.shape, y.shape, n_features)
            # Feature bagging (Select random subsets of the features)
            random_feature_indices = np.random.choice(range(n_features), size=max_features, replace=True)
            # print(random_feature_indices) 
            # Save the features used for training the random forest
            
            # Choose features according to the indices
            sub_X = sub_X[:, random_feature_indices]
            
            self.trees[i].fit(sub_X, sub_y)
        return
    
    def predict(self, X):
        votes = np.zeros((X.shape[0], self.n_estimators))
        for idx, tree in enumerate(self.trees):
            votes[:, idx] = tree.predict(X)
            
        predictions = []
        self.votes = votes
        for pred in votes:
            agg_vote = np.bincount(pred).argmax()
            predictions.append(agg_vote)
        return predictions
        
            
rf = RandomForestClassifier()
rf.fit(X,y)

pred = rf.predict(X)
pred