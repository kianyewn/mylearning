import numpy as np
import math

class DecisionStump:
    def __init__(self):
        #  Determines the prediction. 1 = positive label, -1 = negative label
        self.polarity = 1
        # The index of the feature used for split
        self.feature_index = None
        # The threshold value that the feature should be measured against
        self.threshold = None
        # The amount of say, indicative of the classifier's accuracy
        self.alpha = None
        
class Adabooster:
    def __init__(self, n_stumps=5):
        self.n_stumps = n_stumps
        
    def fit(self,X, y):
        self.clfs = []
        n_samples, n_features = X.shape

        # Initialize equal weights for all samples
        # Sample weights sum to one
        sample_weights = 1/n_samples * np.ones(y.shape[0])
        eps = np.finfo('float').eps

        for _ in range(self.n_stumps):
            stump = DecisionStump()
            min_error = None
            predictions = np.ones(y.shape[0])
            
            # Find the best feature and treshold
            for feature_idx in range(n_features):
                unique_values = np.unique(X[:,feature_idx])
                for threshold in unique_values:
                    polarity = 1
                    predictions = np.ones(y.shape[0])
                    # make predictions according to threshold
                    predictions[X[:,feature_idx] < threshold] = -1
                    
                    # Get the total error according to sample weights
                    # Error will sum to 1 because sample weights sum to 1
                    # error = sum(sample_weights[polarity!=y])
                    error = sum(sample_weights[predictions!=y])
                    # if 50% of the error is classified wrongly
                    # predict the opposite of the current polarity
                    if error > 0.5:
                        polarity = -1
                        # the error when polarity is swapped
                        error = 1 - error
                    
                    if min_error is None or error < min_error:
                        min_error = error
                        stump.feature_index = feature_idx
                        stump.threshold = threshold
                        stump.polarity = polarity
                        
            # Get the amount of say for the stump
            # Smaller min_error -> smaller alpha
            # Larger min_error -> larger alpha
            stump.alpha = 1/2 * math.log((1-min_error + eps) / (min_error +eps))
            
            # Update the sample weights according to amount_of_say
            ####################################################################################
            # Correct classification: (decrease sample weights of correct classifications)
            # -> Future trees Total error will be small even if we make incorrect classification. Good because previous trees already classify it correctly
            # -> Dont keep choosing features that already correctly classify the sample, choose other features that can correctly classify the other examples
            # sample_weight = sample_weight * np.exp(- amount_of_say)
            ####################################################################################
            # Incorrect classification: (increase sample weights of incorrect classification)
            # -> Future trees Total error will be larger for incorrect samples
            # -> Don't choose features that incorrectly classify this example, since total error will increase
            ####################################################################################
            # sample_weight = sample_weight * np.exp(amount_of_say)
           
            predictions  = np.ones(y.shape[0])
            # if polarity is -1, then predict == 1 should be >= threshold
            # if polarity is 1, then prediction ==1 should be < threshold
            negative_idx =(stump.polarity * X[:, stump.feature_index] < stump.polarity * stump.threshold)
            predictions[negative_idx] = -1
            # 1 1 -> 1, -1,-1 = 1
            correctly_classified = y * predictions
            
            sample_weights = sample_weights * np.exp(-stump.alpha * correctly_classified)
            
            # normalize sample weights to 1
            sample_weights = sample_weights / sum(sample_weights)
            self.clfs.append(stump)
            
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, 1))
        
        # for each classifier, label the samples
        for clf in self.clfs:
            # set all predictions to ones iniitally
            predictions = np.ones(np.shape(y_pred))
            # The indexes where sample values are below threshold
            # Also flip label if error > 0.5
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # label those as -1 label
            predictions[negative_idx] = -1
            #Add predictions weighted by the classifiers alpha
            # (alpha indicative of the classifier's proficiency)
            y_pred += clf.alpha * predictions
            
        y_pred = np.sign(y_pred).flatten()
        
        return y_pred
                
            
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
data = datasets.load_digits()
X = data.data
y = data.target

digit1 = 1
digit2 = 8
idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
y = data.target[idx]
# Change labels to {-1, 1}
y[y == digit1] = -1
y[y == digit2] = 1
X = data.data[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Adaboost classification with 5 weak classifiers
clf = Adabooster(n_stumps=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 
y_pred

# (y_pred == y).mean()
# ((y_pred.flatten() == y)==1).mean()
# (y_pred.flatten() == - 1).sum()


accuracy1 = (y_test==y_pred).mean()
accuracy1
print ("Accuracy:", accuracy1)


# import pandas as pd
# pd.Series(y_test).value_counts()



# Decision stump used as weak classifier in this impl. of Adaboost
class DecisionStump():
    def __init__(self):
        # Determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1
        # The index of the feature used to make classification
        self.feature_index = None
        # The threshold value that the feature should be measured against
        self.threshold = None
        # Value indicative of the classifier's accuracy
        self.alpha = None

class Adaboost():
    """Boosting method that uses a number of weak classifiers in 
    ensemble to make a strong classifier. This implementation uses decision
    stumps, which is a one level Decision Tree. 

    Parameters:
    -----------
    n_clf: int
        The number of weak classifiers that will be used. 
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []
        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            # Minimum error given for using a certain feature value threshold
            # for predicting sample label
            min_error = float('inf')
            # Iterate throught every unique feature value and see what value
            # makes the best threshold for predicting y
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # Try every unique feature value as threshold
                for threshold in unique_values:
                    p = 1
                    # Set all predictions to '1' initially
                    prediction = np.ones(np.shape(y))
                    # Label the samples whose values are below threshold as '-1'
                    prediction[X[:, feature_i] < threshold] = -1
                    # Error = sum of weights of misclassified samples
                    error = sum(w[y != prediction])
                    
                    # If the error is over 50% we flip the polarity so that samples that
                    # were classified as 0 are classified as 1, and vice versa
                    # E.g error = 0.8 => (1 - error) = 0.2
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # If this threshold resulted in the smallest error we save the
                    # configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
            # Calculate the alpha which is used to update the sample weights,
            # Alpha is also an approximation of this classifier's proficiency
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1
            # Calculate new weights 
            # Missclassified samples gets larger weights and correctly classified samples smaller
            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # For each classifier => label the samples
        for clf in self.clfs:
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y_pred))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1
            # Add predictions weighted by the classifiers alpha
            # (alpha indicative of classifier's proficiency)
            y_pred += clf.alpha * predictions

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred

        
# # Adaboost classification with 5 weak classifiers
# import math
# clf = Adaboost(n_clf=5)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test) 
# y_pred

# accuracy = (y_test==y_pred).mean()
# print ("Accuracy:", accuracy)