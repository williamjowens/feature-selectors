import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Particle Swarm Optimization feature selector class
class PSOFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_particles=10, n_iterations=100, inertia_weight=0.8, cognitive_weight=1.5,
                 social_weight=1.5, min_features=1, max_features=None, early_stopping_rounds=10,
                 cv=5, random_state=None, verbose=0):
        self.estimator = estimator
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.min_features = min_features
        self.max_features = max_features
        self.early_stopping_rounds = early_stopping_rounds
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], multi_output=True)

        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features_
        elif not 1 <= self.max_features <= self.n_features_:
            raise ValueError("max_features must be between 1 and the number of features")

        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        elif hasattr(X, 'feature_names'):
            self.feature_names_ = X.feature_names.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(self.n_features_)]

        self.particles_ = []
        self.gbest_position_ = None
        self.gbest_fitness_ = -np.inf

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_particles):
            n_features = rng.integers(self.min_features, self.max_features + 1)
            position = rng.choice([0, 1], size=self.n_features_, p=[1 - n_features / self.n_features_, n_features / self.n_features_])
            particle = {
                'position': position,
                'velocity': np.zeros(self.n_features_),
                'pbest_position': position.copy(),
                'pbest_fitness': -np.inf
            }
            self.particles_.append(particle)

        no_improvement_count = 0
        for i in range(self.n_iterations):
            for particle in self.particles_:
                if self.gbest_position_ is None:
                    self.gbest_position_ = particle['position'].copy()
                    self.gbest_fitness_ = self._evaluate_fitness(X[:, particle['position'].astype(bool)], y)

                cognitive_term = self.cognitive_weight * rng.random(self.n_features_) * (particle['pbest_position'] - particle['position'])
                social_term = self.social_weight * rng.random(self.n_features_) * (self.gbest_position_ - particle['position'])
                particle['velocity'] = self.inertia_weight * particle['velocity'] + cognitive_term + social_term
                particle['position'] = np.where(rng.random(self.n_features_) < self._sigmoid(particle['velocity']), 1, 0)

                if np.sum(particle['position']) == 0:
                    particle['position'][rng.integers(0, self.n_features_)] = 1

                fitness = self._evaluate_fitness(X[:, particle['position'].astype(bool)], y)
                if fitness > particle['pbest_fitness']:
                    particle['pbest_position'] = particle['position'].copy()
                    particle['pbest_fitness'] = fitness

                if fitness > self.gbest_fitness_:
                    self.gbest_position_ = particle['position'].copy()
                    self.gbest_fitness_ = fitness
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            if self.verbose > 0:
                print(f"Iteration {i+1}/{self.n_iterations} - Best fitness: {self.gbest_fitness_:.4f}")

            if no_improvement_count >= self.early_stopping_rounds:
                if self.verbose > 0:
                    print(f"Early stopping at iteration {i+1}")
                break

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=['csr', 'csc'])
        if X.shape[1] != self.n_features_:
            raise ValueError("X has a different number of features than during fitting.")

        return X[:, self.gbest_position_.astype(bool)]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _evaluate_fitness(self, X, y):
        cv_scores = cross_val_score(self.estimator, X, y, cv=self.cv)
        return np.mean(cv_scores)

    def get_support(self, indices=False):
        check_is_fitted(self, 'is_fitted_')
        if indices:
            return np.where(self.gbest_position_.astype(bool))[0]
        else:
            return self.gbest_position_.astype(bool)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_
        return np.array(input_features)[self.gbest_position_.astype(bool)]

    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        X_selected = self.transform(X)
        return self.estimator.fit(X_selected, y).score(X_selected, y)

    def get_params(self, deep=True):
        return {
            'estimator': self.estimator,
            'n_particles': self.n_particles,
            'n_iterations': self.n_iterations,
            'inertia_weight': self.inertia_weight,
            'cognitive_weight': self.cognitive_weight,
            'social_weight': self.social_weight,
            'min_features': self.min_features,
            'max_features': self.max_features,
            'early_stopping_rounds': self.early_stopping_rounds,
            'cv': self.cv,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

##################
# Implementation #
##################

if __name__ == '__main__':
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Create an instance of the estimator
    estimator = SVC(kernel='rbf', random_state=42)

    # Create an instance of the PSO feature selector
    pso_selector = PSOFeatureSelector(estimator, n_particles=20, n_iterations=50, min_features=2, max_features=10,
                                      early_stopping_rounds=5, cv=5, random_state=42, verbose=1)

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('feature_selection', pso_selector),
        ('classification', estimator)
    ])

    # Fit the pipeline
    pipeline.fit(X, y)

    # Get information about the selected features
    selected_features = pso_selector.get_support(indices=True)
    selected_feature_names = pso_selector.get_feature_names_out(data.feature_names)
    n_selected_features = np.sum(pso_selector.get_support())

    print("\nFeature Selection Results:")
    print("Number of features selected:", n_selected_features)
    print("Selected feature indices:", selected_features)
    print("Selected feature names:", selected_feature_names)

    # Evaluate the performance of the pipeline
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\nClassification Accuracy: {accuracy:.4f}")