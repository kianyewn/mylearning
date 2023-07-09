import pandas as pd
import numpy as np

class StackingAveragedModels:
    """Class that implements stacking
    """
    def __init__(self, base_models, meta_model, n_folds=5, add_original_features=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.add_original_features = add_original_features

    def get_oof_num(self, X, y):
        kfold = KFold(n_splits = self.n_folds, shuffle=True, random_state=156)
        all_data = pd.concat([X, y], axis=1)
        all_data.loc[:, 'oof_num'] = 0
        for idx, (train_idx, oof_idx) in enumerate(kfold.split(X,y)):
            all_data.loc[oof_idx, 'oof_num'] = idx
        return all_data
    
    def fit(self, X, y):
        self.fitted_base_models = [list() for _ in range(len(self.base_models))]
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        out_of_fold_predictions = pd.DataFrame(out_of_fold_predictions, columns=[f'base_model_{idx+1}_oof_pred' for idx in range(len(self.base_models))])
        oof_metadata = self.get_oof_num(X, y)
        for oof_idx in range(self.n_folds):
            oof_X_train = X[all_data['oof_num'] != oof_idx]
            oof_y_train = y[all_data['oof_num'] != oof_idx]

            oof_X_test = X[all_data['oof_num'] == oof_idx]
            for model_idx, base_model in enumerate(self.base_models):
                fitted_base_model = base_model.fit(oof_X_train, oof_y_train)
                self.fitted_base_models[model_idx].append(fitted_base_model)
                oof_predictions = fitted_base_model.predict_proba(oof_X_test)[:,1]
                out_of_fold_predictions.iloc[oof_X_test.index, model_idx] = oof_predictions
        if self.add_original_features:
            out_of_fold_predictions = pd.concat([X, out_of_fold_predictions], axis=1)
        self.fitted_meta_model = self.meta_model.fit(out_of_fold_predictions, y)
        return self
    
    def predict_proba(self, X):
        meta_features = np.column_stack(
            [np.column_stack([model.predict_proba(X)[:,1] for model in within_fold_models]).mean(axis=1)
             for within_fold_models in self.fitted_base_models])
        
        meta_features = pd.DataFrame(meta_features, columns=[f'base_model_{idx+1}_oof_pred' for idx in range(len(self.base_models))])
        if self.add_original_features:
            meta_features = pd.concat([X, meta_features], axis=1)
        self.meta_features = meta_features
        predictions = self.fitted_meta_model.predict_proba(meta_features)[:,1]
        return predictions
    
# base_models = [model, model, model]
# meta_model = metamodel = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# sam = StackingAveragedModels(base_models, meta_model, add_original_features=True)
# sam_f = sam.fit(X_train, y_train)
# pred = sam.predict_proba(X_train)