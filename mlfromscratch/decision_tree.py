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

def _majority_vote(y):
    most_common = None
    max_count = 0
    for label in np.unique(y):
        # Count number of occurences of samples with label
        count = len(y[y == label])
        if count > max_count:
            most_common = label
            max_count = count
    return most_common
    
y = [1,1,1,0,0]
assert majority_voting(y) == 1
y= [1,0,0,0,1]
assert majority_voting(y) == 0

y = np.array(y)
_majority_vote(y)



from dataclasses import dataclass
from typing import Any
@dataclass
class DecisionNode:
    feature_i: Any = None
    threshold: Any = None
    value: Any = None
    true_branch: Any = None
    false_branch: Any = None
    
# class DecisionNode():
#     def __init__(self,
#                  feature_i=None, 
#                  threshold=None,
#                  value=None, 
#                  true_branch=None,
#                  false_branch=None):
#         self.feature_i = feature_i
#         self.threshold = threshold
#         self.value = value
#         self.true_branch = true_branch
#         self.false_branch = false_branch
        
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



# # Super class of RegressionTree and ClassificationTree
# class DecisionTree(object):
#     """Super class of RegressionTree and ClassificationTree.

#     Parameters:
#     -----------
#     min_samples_split: int
#         The minimum number of samples needed to make a split when building a tree.
#     min_impurity: float
#         The minimum impurity required to split the tree further.
#     max_depth: int
#         The maximum depth of a tree.
#     loss: function
#         Loss function that is used for Gradient Boosting models to calculate impurity.
#     """
#     def __init__(self, min_samples_split=2, min_impurity=1e-7,
#                  max_depth=float("inf"), loss=None):
#         self.root = None  # Root node in dec. tree
#         # Minimum n of samples to justify split
#         self.min_samples_split = min_samples_split
#         # The minimum impurity to justify split
#         self.min_impurity = min_impurity
#         # The maximum depth to grow the tree to
#         self.max_depth = max_depth
#         # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
#         self._impurity_calculation = None
#         # Function to determine prediction of y at leaf
#         self._leaf_value_calculation = None
#         # If y is one-hot encoded (multi-dim) or not (one-dim)
#         self.one_dim = None
#         # If Gradient Boost
#         self.loss = loss

#     def fit(self, X, y, loss=None):
#         """ Build decision tree """
#         self.one_dim = len(np.shape(y)) == 1
#         self.root = self._build_tree(X, y)
#         self.loss=None

#     def _build_tree(self, X, y, current_depth=0):
#         """ Recursive method which builds out the decision tree and splits X and respective y
#         on the feature of X which (based on impurity) best separates the data"""

#         largest_impurity = 0
#         best_criteria = None    # Feature index and threshold
#         best_sets = None        # Subsets of the data

#         # Check if expansion of y is needed
#         if len(np.shape(y)) == 1:
#             y = np.expand_dims(y, axis=1)

#         # Add y as last column of X
#         Xy = np.concatenate((X, y), axis=1)

#         n_samples, n_features = np.shape(X)

#         if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
#             # Calculate the impurity for each feature
#             for feature_i in range(n_features):
#                 # All values of feature_i
#                 feature_values = np.expand_dims(X[:, feature_i], axis=1)
#                 unique_values = np.unique(feature_values)

#                 # Iterate through all unique values of feature column i and
#                 # calculate the impurity
#                 for threshold in unique_values:
#                     # Divide X and y depending on if the feature value of X at index feature_i
#                     # meets the threshold
#                     Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

#                     if len(Xy1) > 0 and len(Xy2) > 0:
#                         # Select the y-values of the two sets
#                         y1 = Xy1[:, n_features:]
#                         y2 = Xy2[:, n_features:]

#                         # Calculate impurity
#                         impurity = self._impurity_calculation(y, y1, y2)

#                         # If this threshold resulted in a higher information gain than previously
#                         # recorded save the threshold value and the feature
#                         # index
#                         if impurity > largest_impurity:
#                             largest_impurity = impurity
#                             best_criteria = {"feature_i": feature_i, "threshold": threshold}
#                             best_sets = {
#                                 "leftX": Xy1[:, :n_features],   # X of left subtree
#                                 "lefty": Xy1[:, n_features:],   # y of left subtree
#                                 "rightX": Xy2[:, :n_features],  # X of right subtree
#                                 "righty": Xy2[:, n_features:]   # y of right subtree
#                                 }

#         if largest_impurity > self.min_impurity:
#             # Build subtrees for the right and left branches
#             true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
#             false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
#             return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
#                                 "threshold"], true_branch=true_branch, false_branch=false_branch)

#         # We're at leaf => determine value
#         leaf_value = self._leaf_value_calculation(y)

#         return DecisionNode(value=leaf_value)


#     def predict_value(self, x, tree=None):
#         """ Do a recursive search down the tree and make a prediction of the data sample by the
#             value of the leaf that we end up at """

#         if tree is None:
#             tree = self.root

#         # If we have a value (i.e we're at a leaf) => return value as the prediction
#         if tree.value is not None:
#             return tree.value

#         # Choose the feature that we will test
#         feature_value = x[tree.feature_i]

#         # Determine if we will follow left or right branch
#         branch = tree.false_branch
#         if isinstance(feature_value, int) or isinstance(feature_value, float):
#             if feature_value >= tree.threshold:
#                 branch = tree.true_branch
#         elif feature_value == tree.threshold:
#             branch = tree.true_branch

#         # Test subtree
#         return self.predict_value(x, branch)

#     def predict(self, X):
#         """ Classify samples one by one and return the set of labels """
#         y_pred = [self.predict_value(sample) for sample in X]
#         return y_pred

#     def print_tree(self, tree=None, indent=" "):
#         """ Recursively print the decision tree """
#         if not tree:
#             tree = self.root

#         # If we're at leaf => print the label
#         if tree.value is not None:
#             print (tree.value)
#         # Go deeper down the tree
#         else:
#             # Print test
#             print ("%s:%s? " % (tree.feature_i, tree.threshold))
#             # Print the true scenario
#             print ("%sT->" % (indent), end="")
#             self.print_tree(tree.true_branch, indent + indent)
#             # Print the false scenario
#             print ("%sF->" % (indent), end="")
#             self.print_tree(tree.false_branch, indent + indent)


# class ClassificationTree(DecisionTree):
#     def _calculate_information_gain(self, y, y1, y2):
#         # Calculate information gain
#         p = len(y1) / len(y)
#         entropy = calculate_entropy(y)
#         info_gain = entropy - p * \
#             calculate_entropy(y1) - (1 - p) * \
#             calculate_entropy(y2)

#         return info_gain

#     def _majority_vote(self, y):
#         most_common = None
#         max_count = 0
#         for label in np.unique(y):
#             # Count number of occurences of samples with label
#             count = len(y[y == label])
#             if count > max_count:
#                 most_common = label
#                 max_count = count
#         return most_common

#     def fit(self, X, y):
#         self._impurity_calculation = self._calculate_information_gain
#         self._leaf_value_calculation = self._majority_vote # majority_vote # self._majority_vote
#         super(ClassificationTree, self).fit(X, y)


# dataset = datasets.load_iris()
# X, y = dataset.data, dataset.target
# y = np.array([1 if yi > 1 else yi for yi in y])
# clf = DecisionTree()
# clf.fit(X, y)

# preds = clf.predict(X)
        
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = datasets.load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = DecisionTree()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print ("Accuracy:", accuracy)
