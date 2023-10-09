from sklearn.base import TransformerMixin

class Calibration(TransformerMixin):
    def __init__(self, nbins=25, degree=1):
        self.nbins = nbins
        self.degree = degree
        
    def prob2lnodds(self, x: float):
        if x == 0:
            # return very negative value, eg -36
            return np.log(np.finfo(float).eps)
        elif x == 1:
            # prevent division by 0
            return np.log(x / (1-x+np.finfo(float).eps))
        else:
            return np.log(x/(1-x))
        
    def lnodds2prob(self, x:float):
        return 1 - 1/(1+np.exp(x))
        
    def fit(self, prob:np.array, label:np.array):
        lst_prob = prob.tolist()
        lst_label = label.tolist()
        
#         # logodds for predicted probability
#         lst_prob_lnodds = [self.prob2lnodds(x) for x in lst_prob]
#         # logodds for true labels
#         lst_label_lnodds = [self.prob2lnodds(x) for x in lst_label]
        
        df_data = pd.DataFrame({'lst_prob': lst_prob,
                               'lst_label': lst_label})
        df_data['bin'] = pd.qcut(df_data['lst_prob'], q=self.nbins, duplicates='drop') 
        df_cal = df_data.groupby('bin').agg(total_count=('lst_prob', 'count'),
                                            expected_percentage=('lst_label', 'mean'),
                                           actual_percentage=('lst_prob', 'mean'))
        # expected perc contains only 1 or 0. -> calculating ln_odds(0) is -inf
        # For bins with zero label == 1, consider just one sample == 1, substituting expected_percentage as 1 / total_users_in_bin
        df_cal['adj_expected_percentage'] = df_cal.apply(lambda x: max(x['expected_percentage'], 1/x['total_count'], 0.0001), axis=1)
        df_cal['ln_odds_expected_percentage'] = df_cal['adj_expected_percentage'].apply(lambda x: self.prob2lnodds(x))
        df_cal['ln_odds_actual_percentage'] = df_cal['actual_percentage'].apply(lambda x: self.prob2lnodds(x))
        
        calibration_coef = np.polyfit(x=df_cal['ln_odds_actual_percentage'], y=df_cal['ln_odds_expected_percentage'], deg=self.degree)
        self.df_cal = df_cal
        self.calibration_coef = calibration_coef
        return 
    
    def transform(self, prob):
        lst_prob = prob.tolist()
        ln_odds_lst_prob = [self.prob2lnodds(x) for x in lst_prob]
        ln_odds_lst_prob_pred = np.poly1d(self.calibration_coef)(ln_odds_lst_prob)
        calibrated_prob = [self.lnodds2prob(x) for x in ln_odds_lst_prob_pred]
        return calibrated_prob
    
#     def transform_inverse(self, calib_prob):
#         ln_odds_lst_prob_pred = [np.prob2lnodds(x) for x in calib_prob]
        
    
if __name__ == '__main__':

    from sklearn import datasets
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier

    data = datasets.load_iris()
    X = data.data
    y = data.target
    y = np.array([1 if yi >= 1 else yi for yi in y])
    X = pd.DataFrame(X, columns=data.feature_names)

    # fit model
    clf = GradientBoostingClassifier()
    clf.fit(X,y)
    # predict
    pred = clf.predict_proba(X)

    # since this is a simple example, predictions are skewed towards 0 and 1
    # add random noise to ditter probabilities for binning
    pred = (pred + abs(np.random.randn(*pred.shape)))
    # Ensure sums to 1
    pred = pred / pred.sum(axis=-1, keepdims=True)
    # do calibration with sigmoid
    calib = Calibration()
    calib.fit(pred[:,1], y)
    calibrated_probabilities = calib.transform(pred[:,1])