import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_is_fitted
from cmaes import CMA

class CMAESFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features_to_select=None, sigma0=1.0, max_iter=50, cv=5, scoring=None, random_state=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.sigma0 = sigma0
        self.max_iter = max_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        self.n_features_to_select = self.n_features_to_select or n_features

        # Initialize the CMA-ES optimizer with the correct dimensions and population size
        es = CMA(mean=np.zeros(n_features), sigma=self.sigma0, seed=self.random_state)
        self.best_score_ = -np.inf
        self.best_mask_ = np.zeros(n_features, dtype=bool)

        for _ in range(self.max_iter):
            solutions = es.ask()
            scores = []

            for solution in solutions:
                indices = np.argsort(solution)[-self.n_features_to_select:]
                mask = np.zeros(n_features, dtype=bool)
                mask[indices] = True
                X_selected = X[:, mask]

                # Ensure X_selected is 2D for compatibility with sklearn estimators
                if X_selected.ndim == 1:
                    X_selected = X_selected.reshape(-1, 1)

                # Calculate the score of the selected features
                score = cross_val_score(self.estimator, X_selected, y, cv=self.cv, scoring=self.scoring).mean()
                scores.append(score)

                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_mask_ = mask

            # The tell method should be given a list of tuples, each containing a solution and its score
            es.tell([(solutions[i], -scores[i]) for i in range(len(solutions))])

        return self

    def transform(self, X):
        check_is_fitted(self, 'best_mask_')
        return X[:, self.best_mask_]

    def predict(self, X):
        check_is_fitted(self, 'best_mask_')
        X_selected = self.transform(X)
        self.estimator.fit(self.X_[:, self.best_mask_], self.y_)
        return self.estimator.predict(X_selected)

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier

    data = load_digits()
    X, y = data.data, data.target

    estimator = RandomForestClassifier(random_state=42)
    selector = CMAESFeatureSelector(estimator=estimator, sigma0=0.5, max_iter=100, random_state=42)
    
    selector.fit(X, y)
    X_transformed = selector.transform(X)

    print(f"Selected features shape: {X_transformed.shape}")
    print(f"Best score achieved: {selector.best_score_}")