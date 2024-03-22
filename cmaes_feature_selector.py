import numpy as np
import cma
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import get_scorer
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# CMA-ES feature selector class
class CMAESFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, sigma=0.5, n_population=50, n_generations=100,
                 scoring=None, cv=5, verbose=0, random_state=None, early_stopping=10,
                 n_jobs=None, cma_options=None, min_features=1, penalty_factor=0.1):
        self.estimator = estimator
        self.sigma = sigma
        self.n_population = n_population
        self.n_generations = n_generations
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_jobs = n_jobs
        self.cma_options = cma_options
        self.min_features = min_features
        self.penalty_factor = penalty_factor
        self.scaler = StandardScaler()

    def _evaluate_fitness(self, individual, X, y):
        mask = individual[:-1] >= 0.5
        n_selected_features = np.sum(mask)
        
        if n_selected_features < self.min_features:
            penalty = self.penalty_factor
        else:
            penalty = 1.0
        
        if n_selected_features == 0:
            return -np.inf
        
        X_subset = X[:, mask]
        scorer = get_scorer(self.scoring)
        scores = cross_val_score(self.estimator, X_subset, y, cv=self.cv, scoring=scorer,
                                 n_jobs=self.n_jobs)
        return np.mean(scores) * penalty

    def _cmaes_optimization(self, X, y):
        n_features = X.shape[1]
        mean = np.zeros(n_features + 1)
        mean[-1] = 0.5
        cov = np.eye(n_features + 1)

        if self.cma_options is None:
            cma_options = {'popsize': self.n_population, 'verb_disp': self.verbose,
                           'bounds': [0, 1], 'seed': self.random_state,
                           'verb_log': 0, 'verbose': -9}
        else:
            cma_options = self.cma_options

        if self.verbose > 0:
            print("CMA-ES options:", cma_options)

        es = cma.CMAEvolutionStrategy(mean, self.sigma, inopts=cma_options)

        best_fitness = -np.inf
        best_individual = None
        n_generations_without_improvement = 0

        for generation in range(self.n_generations):
            population = es.ask()
            population = np.array(population)
            population[:, -1] = np.clip(population[:, -1], 0, 1)

            population[:, :-1] = np.where(population[:, :-1] < 0.5, 0, 1)
            feature_counts = np.sum(population[:, :-1], axis=1)
            population[feature_counts < self.min_features, :-1] = 1

            fitness_values = [self._evaluate_fitness(individual, X, y) for individual in population]

            es.tell(population, [-score for score in fitness_values])
            es.logger.add()

            best_index = np.argmax(fitness_values)
            best_individual_in_generation = population[best_index]

            if fitness_values[best_index] > best_fitness:
                best_fitness = fitness_values[best_index]
                best_individual = best_individual_in_generation
                n_generations_without_improvement = 0
            else:
                n_generations_without_improvement += 1

            if self.verbose > 0 and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

            if n_generations_without_improvement >= self.early_stopping:
                if self.verbose > 0:
                    print(f"Early stopping at generation {generation}")
                break

        self.best_individual_ = best_individual
        self.best_fitness_ = best_fitness
        self.es_ = es

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        X = self.scaler.fit_transform(X)

        if self.verbose > 0:
            print("Selecting optimal features using CMA-ES")

        self._cmaes_optimization(X, y)

        self.support_ = self.best_individual_[:-1] >= 0.5
        self.n_features_ = int(np.sum(self.support_))
        self.feature_names_ = np.array(range(X.shape[1]))[self.support_]

        if self.verbose > 0:
            print(f"Best individual: {self.best_individual_}")
            print(f"Best fitness: {self.best_fitness_}")
            print(f"Number of selected features: {self.n_features_}")

        return self

    def transform(self, X):
        check_is_fitted(self, ['support_', 'n_features_', 'scaler'])
        X = check_array(X, dtype=None)
        X = self.scaler.transform(X)
        return X[:, self.support_][:, :self.n_features_]

    def _more_tags(self):
        return {'requires_y': True}

    def get_support(self, indices=False):
        check_is_fitted(self, ['support_', 'feature_names_'])
        return self.support_ if not indices else self.feature_names_

##################
# Implementation #
##################

if __name__ == '__main__':
    # Load the benchmark dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the estimator
    estimator = SVC(kernel='rbf', random_state=42)

    # Create an instance of the CMAESFeatureSelector
    selector = CMAESFeatureSelector(estimator=estimator, n_generations=50, n_population=30,
                                    scoring='accuracy', cv=5, verbose=1, random_state=42,
                                    early_stopping=5, n_jobs=-1)

    # Create a pipeline with the feature selector and the estimator
    pipeline = Pipeline([
        ('selector', selector),
        ('estimator', estimator)
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Evaluate the performance on the testing data
    accuracy = pipeline.score(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Get the selected feature names
    selected_feature_names = feature_names[selector.get_support(indices=True)]
    print(f"\nSelected Features: {selected_feature_names}")