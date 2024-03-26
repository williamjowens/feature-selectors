import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.feature_selection import mutual_info_classif

# Game Theory Optimization feature selector class
class GTOFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, payoff_function=None, equilibrium_concept='nash', max_iterations=100, min_features=1, step=1, scoring=None, cv=5, random_state=None):
        self.estimator = estimator
        self.payoff_function = payoff_function
        self.equilibrium_concept = equilibrium_concept
        self.max_iterations = max_iterations
        self.min_features = min_features
        self.step = step
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.selected_features_ = None
        self.selected_feature_names_ = None
        self.best_score_ = None
        self.best_n_features_ = None

    def fit(self, X, y=None, feature_names=None):
        # Check if feature names are provided
        if feature_names is None:
            feature_names = ['Feature ' + str(i+1) for i in range(X.shape[1])]

        # Set the maximum number of features
        max_features = min(X.shape[1], len(feature_names))

        # Set the default payoff function if not provided
        if self.payoff_function is None:
            self.payoff_function = self._mutual_info_payoff

        # Set the default scoring function if not provided
        if self.scoring is None:
            self.scoring = make_scorer(accuracy_score)

        self.best_score_ = -np.inf
        self.best_n_features_ = self.min_features

        for n_features in range(self.min_features, max_features + 1, self.step):
            # Initialize players' strategies randomly
            strategies = self._initialize_strategies(X.shape[1], n_features)

            for _ in range(self.max_iterations):
                # Compute payoffs for each player
                payoffs = self._compute_payoffs(X, y, strategies)

                # Update players' strategies based on the equilibrium concept
                strategies = self._update_strategies(strategies, payoffs)

                # Check for convergence or termination condition
                if self._check_convergence(strategies):
                    break

            # Select the features based on the final strategies
            selected_features = self._select_features(strategies)
            selected_feature_names = [feature_names[i] for i in selected_features]

            # Evaluate the selected features using cross-validation
            X_selected = X[:, selected_features]
            scores = cross_val_score(self.estimator, X_selected, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(scores)

            # Update the best score, best number of features, and selected feature names
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_n_features_ = n_features
                self.selected_features_ = selected_features
                self.selected_feature_names_ = selected_feature_names

        return self

    def transform(self, X):
        # Return the selected features
        if self.selected_features_ is not None:
            return X[:, self.selected_features_]
        else:
            return X

    def _initialize_strategies(self, n_features, n_features_to_select):
        # Initialize players' strategies randomly
        strategies = np.zeros((n_features, n_features_to_select))
        for i in range(n_features):
            indices = np.random.choice(n_features_to_select, size=n_features_to_select, replace=False)
            strategies[i, indices] = 1
        return strategies

    def _compute_payoffs(self, X, y, strategies):
        # Compute payoffs for each player based on the payoff function
        payoffs = np.zeros(strategies.shape)
        for i in range(strategies.shape[0]):
            for j in range(strategies.shape[1]):
                if strategies[i, j] == 1:
                    subset_mask = strategies[:, j] == 1
                    X_subset = X[:, subset_mask]
                    payoffs[i, j] = self.payoff_function(X_subset, y)
        return payoffs

    def _update_strategies(self, strategies, payoffs):
        # Update players' strategies based on the equilibrium concept
        if self.equilibrium_concept == 'nash':
            # Implement Nash equilibrium update rule
            avg_payoffs = np.mean(payoffs, axis=1, keepdims=True)
            strategies *= payoffs / avg_payoffs
            strategies /= np.sum(strategies, axis=1, keepdims=True)
        return strategies

    def _check_convergence(self, strategies, tolerance=1e-4):
        # Check for convergence or termination condition
        diff = np.max(np.abs(strategies - np.mean(strategies, axis=0, keepdims=True)))
        return diff < tolerance

    def _select_features(self, strategies):
        # Select the features based on the final strategies
        selected_features = np.where(np.sum(strategies, axis=0) > 0)[0]
        return selected_features

    def _mutual_info_payoff(self, X, y):
        # Compute mutual information between features and target variable
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        return np.mean(mi)

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline

    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    # Create an instance of GTOFeatureSelector with a decision tree classifier
    selector = GTOFeatureSelector(estimator=DecisionTreeClassifier(random_state=42), min_features=1, step=1, cv=5, random_state=42)

    # Create a pipeline with feature selection and the specified classifier
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('classifier', selector.estimator)
    ])

    # Evaluate the pipeline using cross-validation
    accuracy_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    f1_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
    mean_accuracy = np.mean(accuracy_scores)
    mean_f1 = np.mean(f1_scores)

    # Fit the selector with feature names
    selector.fit(X, y, feature_names=feature_names)

    # Print the selected features, mean accuracy score, and mean F1 score
    print("Selected features:", selector.selected_feature_names_)
    print("Mean accuracy score:", mean_accuracy)
    print("Mean F1 score:", mean_f1)