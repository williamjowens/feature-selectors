import numpy as np
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Branch & Cut feature selector class
class BranchAndCutFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, objectives, constraints, min_features=1, max_features=None, branching_strategy='most_fractional',
                 preprocessing=StandardScaler(), warm_start=False, verbose=False, max_iter=1000, tol=1e-6):
        self.objectives = objectives
        self.constraints = constraints
        self.min_features = min_features
        self.max_features = max_features
        self.branching_strategy = branching_strategy
        self.preprocessing = preprocessing
        self.warm_start = warm_start
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.best_solution_ = None
        self.best_score_ = -np.inf
        self.selected_features_ = None

    def fit(self, X, y=None):
        X = self._preprocess(X)
        num_features = X.shape[1]
        self.max_features_ = self.max_features if self.max_features is not None else num_features

        if self.warm_start and self.best_solution_ is not None:
            current_solution = self.best_solution_
        else:
            current_solution = np.zeros(num_features)

        self._branch_and_cut(X, y, current_solution, 0)

        if self.best_solution_ is None:
            self.selected_features_ = np.arange(num_features)
        else:
            self.selected_features_ = np.where(self.best_solution_ >= self.tol)[0]

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self._preprocess(X)
        return X[:, self.selected_features_]

    def _branch_and_cut(self, X, y, current_solution, current_depth):
        if current_depth >= self.max_iter:
            return

        if self._is_integer(current_solution):
            current_obj_value = self._evaluate_objective(X, y, current_solution)
            if current_obj_value > self.best_score_ and self._is_feasible(current_solution):
                self.best_solution_ = current_solution
                self.best_score_ = current_obj_value
                if self.verbose:
                    print(f"Depth: {current_depth}, Objective: {current_obj_value}, Solution: {current_solution}")
            return

        lp_solution = self._solve_lp_relaxation(X, y, current_solution)

        if lp_solution is None:
            return

        if self._is_integer(lp_solution):
            current_obj_value = self._evaluate_objective(X, y, lp_solution)
            if current_obj_value > self.best_score_ and self._is_feasible(lp_solution):
                self.best_solution_ = lp_solution
                self.best_score_ = current_obj_value
                if self.verbose:
                    print(f"Depth: {current_depth}, Objective: {current_obj_value}, Solution: {lp_solution}")
            return

        cutting_planes = self._cutting_planes(X, y, lp_solution)
        if len(cutting_planes) > 0:
            self.constraints.extend(cutting_planes)
            self._branch_and_cut(X, y, current_solution, current_depth)
            self.constraints = self.constraints[:-len(cutting_planes)]
            return

        primal_solution = self._primal_heuristic(X, y, lp_solution)
        if primal_solution is not None:
            current_obj_value = self._evaluate_objective(X, y, primal_solution)
            if current_obj_value > self.best_score_ and self._is_feasible(primal_solution):
                self.best_solution_ = primal_solution
                self.best_score_ = current_obj_value
                if self.verbose:
                    print(f"Depth: {current_depth}, Objective: {current_obj_value}, Solution: {primal_solution}")

        branching_var = self._select_branching_variable(lp_solution)
        if branching_var is not None:
            left_solution = np.copy(current_solution)
            left_solution[branching_var] = 0
            right_solution = np.copy(current_solution)
            right_solution[branching_var] = 1

            self._branch_and_cut(X, y, left_solution, current_depth + 1)
            self._branch_and_cut(X, y, right_solution, current_depth + 1)

    def _solve_lp_relaxation(self, X, y, current_solution):
        num_constraints = len(self.constraints) + 2
        num_vars = X.shape[1]

        A_ub = np.zeros((num_constraints, num_vars))
        b_ub = np.zeros(num_constraints)

        for i, (coeffs, rhs) in enumerate(self.constraints):
            A_ub[i] = coeffs
            b_ub[i] = rhs

        A_ub[-2] = np.ones(num_vars)
        b_ub[-2] = self.max_features_
        A_ub[-1] = -np.ones(num_vars)
        b_ub[-1] = -self.min_features

        bounds = [(0, 1) if current_solution[i] == 0 else (current_solution[i], current_solution[i]) for i in range(num_vars)]

        c = -self._evaluate_objective_gradient(X, y, current_solution)
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex', options={'tol': self.tol})

        if result.success:
            return result.x
        else:
            return None

    def _is_integer(self, solution):
        return np.all(np.isclose(solution, np.round(solution), atol=self.tol))

    def _is_feasible(self, solution):
        num_selected = np.sum(solution >= self.tol)
        return self.min_features <= num_selected <= self.max_features_

    def _select_branching_variable(self, solution):
        if self.branching_strategy == 'most_fractional':
            fractional_vars = np.where(~np.isclose(solution, np.round(solution), atol=self.tol))[0]
            if len(fractional_vars) > 0:
                fractions = np.abs(solution[fractional_vars] - np.round(solution[fractional_vars]))
                return fractional_vars[np.argmax(fractions)]
        elif self.branching_strategy == 'random':
            fractional_vars = np.where(~np.isclose(solution, np.round(solution), atol=self.tol))[0]
            if len(fractional_vars) > 0:
                return np.random.choice(fractional_vars)
        return None

    def _cutting_planes(self, X, y, solution):
        cutting_planes = []

        # Add a cutting plane based on feature correlation
        corr_matrix = np.corrcoef(X.T)
        corr_threshold = 0.8
        highly_correlated = np.abs(corr_matrix) >= corr_threshold
        for i in range(X.shape[1]):
            if solution[i] > self.tol:
                for j in range(i+1, X.shape[1]):
                    if highly_correlated[i, j] and solution[j] > self.tol:
                        cutting_planes.append((np.array([1, 1]), 1))

        return cutting_planes

    def _primal_heuristic(self, X, y, solution):
        # Rounding heuristic
        rounded_solution = np.round(solution)
        if self._is_feasible(rounded_solution):
            return rounded_solution
        else:
            return None

    def _preprocess(self, X):
        if self.preprocessing is not None:
            return self.preprocessing.fit_transform(X)
        else:
            return X

    def _evaluate_objective(self, X, y, solution):
        selected_features = np.where(solution >= self.tol)[0]
        obj_value = np.sum(self.objectives[selected_features])
        return obj_value

    def _evaluate_objective_gradient(self, X, y, solution):
        return self.objectives

if __name__ == "__main__":
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the objectives and constraints for feature selection
    num_features = X_train.shape[1]
    objectives = -np.abs(np.corrcoef(X_train.T, y_train)[:-1, -1])  # Negative correlation with target variable
    constraints = [
        (np.ones(num_features), num_features),  # Constraint: Select at most all features
        (-np.ones(num_features), -1)  # Constraint: Select at least one feature
    ]

    # Create an instance of the BranchAndCutFeatureSelector
    selector = BranchAndCutFeatureSelector(objectives, constraints, min_features=5, max_features=10,
                                           branching_strategy='most_fractional', preprocessing=StandardScaler(),
                                           warm_start=True, verbose=True, max_iter=1000, tol=1e-6)

    # Run the feature selection
    selector.fit(X_train, y_train)
    selected_feature_indices = selector.selected_features_
    selected_feature_names = feature_names[selected_feature_indices]

    print("Selected Feature Indices:")
    print(selected_feature_indices)
    print("Selected Feature Names:")
    print(selected_feature_names)

    # Train an SVM classifier using the selected features
    svm = SVC()
    svm.fit(X_train[:, selected_feature_indices], y_train)

    # Evaluate the classifier on the testing set
    y_pred = svm.predict(X_test[:, selected_feature_indices])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")