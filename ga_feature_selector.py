import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# Genetic Algorithm feature selector class
class GeneticAlgorithmFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features=None, population_size=100, n_generations=100, 
                 mutation_rate=0.01, crossover_rate=0.8, tournament_size=5, elitism_rate=0.1,
                 cv=5, scoring='accuracy', n_jobs=None, verbose=0, random_state=None,
                 early_stopping_patience=10):
        self.estimator = estimator
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.best_individual_ = None
        self.best_fitness_ = -np.inf

    def _initialize_population(self, n_features):
        population = np.random.randint(0, 2, size=(self.population_size, n_features))
        population = np.where(population.sum(axis=1, keepdims=True) == 0, 1, population)
        return population

    def _evaluate_fitness(self, individual, X, y):
        selected_features = X[:, individual == 1]
        
        if selected_features.shape[1] == 0:
            return -np.inf
        
        scores = cross_val_score(self.estimator, selected_features, y, cv=self.cv, 
                                 scoring=self.scoring, n_jobs=self.n_jobs)
        return np.mean(scores)

    def _selection(self, population, fitness_scores):
        selected_indices = np.zeros(len(population), dtype=int)
        for i in range(len(population)):
            tournament_indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices[i] = winner_index
        return population[selected_indices]

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1, parent2

    def _mutation(self, individual):
        mutation_mask = np.random.rand(len(individual)) < self.mutation_rate
        individual[mutation_mask] = 1 - individual[mutation_mask]
        
        if individual.sum() == 0:
            individual[np.random.randint(len(individual))] = 1
        return individual

    def fit(self, X, y, feature_names=None):
        self.n_features = X.shape[1]

        if feature_names is None:
            feature_names = [str(i) for i in range(self.n_features)]
        self.feature_names_ = feature_names

        population = self._initialize_population(self.n_features)
        rng = np.random.default_rng(self.random_state)

        early_stopping_counter = 0

        for generation in range(self.n_generations):
            fitness_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_fitness)(individual, X, y) for individual in population
            )
            fitness_scores = np.array(fitness_scores)

            best_fitness_current = np.max(fitness_scores)

            if best_fitness_current > self.best_fitness_:
                self.best_individual_ = population[np.argmax(fitness_scores)]
                self.best_fitness_ = best_fitness_current
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if self.verbose > 0 and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness_:.3f}")

            if early_stopping_counter >= self.early_stopping_patience:
                if self.verbose > 0:
                    print(f"Early stopping at generation {generation}")
                break

            parents = self._selection(population, fitness_scores)
            offspring = []
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self._crossover(parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)

            for i in range(len(offspring)):
                if rng.random() < self.mutation_rate:
                    offspring[i] = self._mutation(offspring[i])

            elite_size = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite_individuals = population[elite_indices]

            offspring_size = self.population_size - elite_size
            offspring_indices = np.random.choice(len(offspring), size=offspring_size, replace=False)
            offspring = np.array(offspring)[offspring_indices]

            population = np.concatenate((elite_individuals, offspring))

        return self

    def transform(self, X):
        selected_features = X[:, self.best_individual_ == 1]
        selected_feature_names = [name for name, selected in zip(self.feature_names_, self.best_individual_) if selected]
        return selected_features, selected_feature_names

##################
# Implementation #
##################

if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the estimator
    estimator = SVC(kernel='rbf', C=1, random_state=42)

    # Create an instance of GeneticAlgorithmFeatureSelection
    ga_selector = GeneticAlgorithmFeatureSelection(
        estimator=estimator,
        n_features=2,
        population_size=50,
        n_generations=100,
        mutation_rate=0.01,
        crossover_rate=0.8,
        tournament_size=5,
        elitism_rate=0.1,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        early_stopping_patience=10
    )

    # Fit the feature selector
    ga_selector.fit(X_train, y_train, feature_names)

    # Transform the training and testing data
    X_train_selected, selected_feature_names = ga_selector.transform(X_train)
    X_test_selected, _ = ga_selector.transform(X_test)

    # Train the estimator on the selected features
    estimator.fit(X_train_selected, y_train)

    # Evaluate the estimator on the testing data
    y_pred = estimator.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)

    print("Selected Features:", selected_feature_names)
    print("Number of Features Selected:", len(selected_feature_names))
    print("Best Fitness Score:", ga_selector.best_fitness_)
    print("Test Accuracy:", accuracy)