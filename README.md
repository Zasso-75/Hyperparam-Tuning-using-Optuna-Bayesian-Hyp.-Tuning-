# Hyperparameter Tuning with Optuna, Grid Search, and Random Search

This project demonstrates efficient hyperparameter tuning of a **Random Forest Classifier** using three different methods:

*  **Bayesian Optimization with Optuna**
*  **Grid Search (exhaustive)**
*  **Random Search (random sampling)**

---

##  Optuna with Bayesian Optimization

###  What is Bayesian Optimization?

Bayesian optimization is a smart technique for optimizing expensive functions (like model training). It uses a **surrogate model** to approximate the objective function and an **acquisition function** to decide the next best set of parameters to evaluate.

* **Surrogate Function**: A probability model (like Tree-structured Parzen Estimator — TPE in Optuna) that approximates the unknown objective function.
* **Acquisition Function**: Selects the most promising hyperparameter values based on the surrogate model by balancing exploration and exploitation.

---

###  Objective Function

We defined an `objective(trial)` function for Optuna where it:

* Suggests values for hyperparameters
* Trains a `RandomForestClassifier`
* Returns the validation accuracy (which Optuna tries to maximize)

```python
params = {
    'n_estimators' : trial.suggest_int('n_estimators', 100, 1000, step=100),
    'max_depth' : trial.suggest_int('max_depth', 3, 15),  # selects int in [3, 15]
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2'])
}
```

Optuna explored the space using its internal **TPE-based Bayesian optimization**, and:

 Found the best parameters in just **1 minute 13 seconds**.

---

##  Comparison with Other Methods

###  Grid Search

```python
param_grid = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
```

* Performed **4,000 fits**
* Took **11 minutes**
* Found **same result** as Optuna, but much slower

---

###  Random Search

```python
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}
```

* Performed **500 fits**
* Faster than Grid Search
* Less precise, but often acceptable

---



##  Conclusion

Optuna with Bayesian optimization (TPE) provided a **faster and smarter** approach to hyperparameter tuning, significantly reducing training time while achieving the same or better performance than traditional grid/random search methods.
