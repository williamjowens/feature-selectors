import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Conjugate Gradient feature selector class
class ConjugateGradientFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, tol=1e-6, max_iter=1000, alpha=0.1, lambda_reg=0.1,
                 cv=5, early_stopping=True, early_stopping_rounds=10, scoring='r2',
                 min_features_to_select=1):
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.cv = cv
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self._feature_names = None

    def _proximal_operator_l1(self, w, threshold):
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

    def _conjugate_gradient(self, X, y, w):
        r = X.T @ (X @ w - y)
        p = -r
        r_norm_sq = r.T @ r

        for _ in range(self.max_iter):
            Ap = X.T @ (X @ p)
            alpha = r_norm_sq / (p.T @ Ap + self.tol)
            w += alpha * p
            r += alpha * Ap
            r_norm_sq_new = r.T @ r

            if np.sqrt(r_norm_sq_new) < self.tol:
                break

            beta = r_norm_sq_new / (r_norm_sq + self.tol)
            p = -r + beta * p
            r_norm_sq = r_norm_sq_new

        return w

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=[np.float64, np.float32], y_numeric=True, multi_output=False)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self._feature_names is not None:
            if len(self._feature_names) != n_features:
                raise ValueError("Number of feature names does not match the number of features.")
            self.feature_names_in_ = np.asarray(self._feature_names)
        else:
            self.feature_names_in_ = [f'feature_{i}' for i in range(n_features)]

        w = np.zeros((n_features, 1))

        best_score = -np.inf
        best_w = None
        best_n_features = 0
        no_improvement_count = 0

        for _ in range(self.max_iter):
            w = self._conjugate_gradient(X, y.reshape(-1, 1), w)
            w = self._proximal_operator_l1(w, self.alpha * self.lambda_reg)

            selected_features = np.abs(w) > self.tol
            n_selected_features = selected_features.sum()

            if n_selected_features >= self.min_features_to_select:
                X_selected = X[:, selected_features.flatten()]
                scores = cross_val_score(LinearRegression(), X_selected, y, cv=self.cv, scoring=self.scoring)
                mean_score = np.mean(scores)

                if mean_score > best_score:
                    best_score = mean_score
                    best_w = w
                    best_n_features = n_selected_features
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if self.early_stopping and no_improvement_count >= self.early_stopping_rounds:
                        break
            else:
                no_improvement_count += 1
                if self.early_stopping and no_improvement_count >= self.early_stopping_rounds:
                    break

        if best_w is None:
            raise ValueError("No features were selected. Try adjusting the selection parameters.")

        self.feature_importances_ = np.abs(best_w).flatten()
        ranking = np.argsort(self.feature_importances_)[::-1]
        self.selected_features_ = ranking[:best_n_features]
        self.n_features_out_ = best_n_features

        return self

    def transform(self, X):
        check_is_fitted(self, ['selected_features_', 'n_features_in_'])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input has a different number of features than the one used during fitting.")

        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        check_is_fitted(self, ['selected_features_', 'n_features_in_'])

        if indices:
            return self.selected_features_
        else:
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[self.selected_features_] = True
            return mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['selected_features_'])
        
        if input_features is None:
            input_features = self.feature_names_in_
        
        if len(input_features) != self.n_features_in_:
            raise ValueError("Number of input features does not match the number of features used during fitting.")

        selected_features_mask = self.get_support()
        return np.array(input_features)[selected_features_mask]
    
    def set_feature_names_in(self, feature_names):
        self._feature_names = feature_names

##################
# Implementation #
##################

if __name__ == '__main__':
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create an instance of the conjugate gradient feature selector
    selector = ConjugateGradientFeatureSelector(tol=1e-4, max_iter=1000, alpha=0.01, lambda_reg=0.1,
                                                 cv=5, early_stopping=True, early_stopping_rounds=10,
                                                 scoring='r2', min_features_to_select=1)

    # Set the feature names in the selector
    selector.set_feature_names_in(feature_names)

    # Fit the feature selector on the training data
    selector.fit(X_train_scaled, y_train)

    # Transform the training and testing data using the selected features
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    # Train a ridge regression model on the selected features
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_selected, y_train)

    # Evaluate the model on the testing data
    y_pred = ridge.predict(X_test_selected)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("Number of selected features:", selector.n_features_out_)
    print("Selected features:", selector.get_feature_names_out())
    print("R-squared score:", r2)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)