    check_is_fitted(self, ['best_mask_', 'n_features_'])
        X = check_array(X)
        return X[:, self.best_mask_]