import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated Annealing feature selector class
class SimulatedAnnealingFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, initial_temp=100, final_temp=0.1, cooldown_factor=0.9,
                 n_iterations=1000, step_size=1, scoring='accuracy', cv=5, random_state=None,
                 verbose=0):
        self.estimator = estimator
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooldown_factor = cooldown_factor
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def _generate_initial_state(self, n_features):
        return np.random.choice([True, False], size=n_features)

    def _calculate_energy(self, state, X, y):
        X_subset = X[:, state]
        scores = cross_val_score(self.estimator, X_subset, y, cv=self.cv, scoring=self.scoring)
        return -np.mean(scores)

    def _perturb_state(self, state):
        new_state = state.copy()
        for _ in range(self.step_size):
            idx = np.random.randint(len(new_state))
            new_state[idx] = not new_state[idx]
        return new_state

    def _simulated_annealing(self, X, y):
        n_features = X.shape[1]
        current_state = self._generate_initial_state(n_features)
        current_energy = self._calculate_energy(current_state, X, y)
        best_state = current_state
        best_energy = current_energy
        best_iteration = 0
        patience = 50

        temp = self.initial_temp

        for i in range(self.n_iterations):
            new_state = self._perturb_state(current_state)
            new_energy = self._calculate_energy(new_state, X, y)

            if new_energy < current_energy:
                current_state = new_state
                current_energy = new_energy
                if current_energy < best_energy:
                    best_state = current_state
                    best_energy = current_energy
                    best_iteration = i
            else:
                acceptance_prob = np.exp((current_energy - new_energy) / temp)
                if np.random.rand() < acceptance_prob:
                    current_state = new_state
                    current_energy = new_energy

            temp = max(temp * self.cooldown_factor, self.final_temp)

            if self.verbose > 0 and (i + 1) % (self.n_iterations // 10) == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Best energy: {best_energy:.4f}")

            if i - best_iteration >= patience:
                if self.verbose > 0:
                    print(f"Early stopping at iteration {i + 1}")
                break

        self.best_state_ = best_state
        self.best_energy_ = best_energy
        self.n_features_selected_ = np.sum(best_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._simulated_annealing(X, y)
        self.is_fitted_ = True

        if self.verbose > 0:
            print(f"Best energy: {self.best_energy_:.4f}")
            print(f"Number of selected features: {self.n_features_selected_}")

        return self

    def transform(self, X):
        check_is_fitted(self, 'is_fitted_')
        return X[:, self.best_state_]

    def _more_tags(self):
        return {'requires_y': True}
    
##################
# Implementation #
##################

if __name__ == '__main__':
    # Load the benchmark dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the base estimator
    base_estimator = SVC(kernel='rbf', random_state=42)

    # Create an instance of the SimulatedAnnealingFeatureSelector
    selector = SimulatedAnnealingFeatureSelector(estimator=base_estimator, n_iterations=150, step_size=2,
                                                 scoring='accuracy', cv=5, random_state=42, verbose=1)

    # Fit the feature selector on the training data
    selector.fit(X_train, y_train)

    # Transform the training and testing data using the selected features
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Train the base estimator on the selected features
    base_estimator.fit(X_train_selected, y_train)

    # Evaluate the performance on the testing data
    y_pred = base_estimator.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Print the selected features
    selected_feature_indices = np.where(selector.best_state_)[0]
    selected_features = feature_names[selected_feature_indices]
    print(f"\nSelected Features: {', '.join(selected_features)}")