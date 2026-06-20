# hyperopt-framework

A lightweight hyperparameter optimization library built from scratch in Python. Plug in any model, define a search space, pick a strategy, and automatically run, track, and compare experiments — no Optuna or Ray Tune under the hood.

## Features

- **Search Strategies** — Random Search, Grid Search, Bayesian Optimization (Gaussian Process + Expected Improvement), and Hyperband (successive halving with bracket management)
- **Flexible Search Spaces** — Uniform, LogUniform, IntUniform, and Categorical hyperparameter types
- **Experiment Tracking** — SQLite-backed tracker that logs every trial with params, scores, metrics, and timing
- **Pluggable Architecture** — implement `SearchStrategy` to add your own algorithms; any callable works as an objective

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from hyperopt import (
    SearchSpace, LogUniform, Uniform, Categorical,
    RandomSearch, Optimizer, ObjectiveDirection, SQLiteTracker,
)

# Define your search space
space = SearchSpace()
space.add(LogUniform("learning_rate", 1e-5, 1e-1))
space.add(Uniform("dropout", 0.0, 0.9))
space.add(Categorical("optimizer", ["sgd", "adam", "rmsprop"]))

# Define your objective (any callable that takes params and returns a score)
def train_model(params):
    # ... train your model here ...
    return validation_loss

# Run optimization
strategy = RandomSearch(space, seed=42)
with SQLiteTracker("experiments.db", experiment_name="my_experiment") as tracker:
    opt = Optimizer(
        strategy=strategy,
        objective=train_model,
        direction=ObjectiveDirection.MINIMIZE,
        tracker=tracker,
    )
    best = opt.run(n_trials=100)

print(f"Best score: {best.score}")
print(f"Best params: {best.params}")
```

## Strategies

### Random Search

Samples uniformly from the search space. Good baseline.

```python
from hyperopt import RandomSearch
strategy = RandomSearch(space, seed=42)
```

### Grid Search

Exhaustive search over a discretized grid. Set `resolution` to control how many points per continuous dimension.

```python
from hyperopt import GridSearch
strategy = GridSearch(space, resolution=10)
print(f"Total combinations: {strategy.total_combinations}")
```

### Bayesian Optimization

Gaussian Process surrogate with Expected Improvement acquisition function. Normalizes inputs and targets, uses Cholesky decomposition for numerical stability.

```python
from hyperopt import BayesianOptimization
strategy = BayesianOptimization(
    space,
    seed=42,
    n_initial=10,       # random trials before GP kicks in
    n_candidates=1000,  # candidate points to evaluate EI over
)
```

### Hyperband

Adaptive resource allocation via successive halving. Requires a `report_fn(params, resource) -> score` where `resource` controls training budget (epochs, data fraction, etc.).

```python
from hyperopt import Hyperband

def report_fn(params, resource):
    # resource = number of epochs to train
    model = build_model(params)
    model.fit(X_train, y_train, epochs=resource)
    return model.evaluate(X_val, y_val)

hb = Hyperband(
    space,
    report_fn=report_fn,
    max_resource=81,  # max epochs
    eta=3,            # reduction factor
    seed=42,
)
best = hb.run_all()
```

## Experiment Tracking

All trials are logged to SQLite with full metadata:

```python
with SQLiteTracker("experiments.db", experiment_name="v1") as tracker:
    optimizer = Optimizer(strategy, objective, tracker=tracker)
    optimizer.run(n_trials=50)

    # Query results
    all_trials = tracker.get_all_trials()
    best = tracker.get_best_trial(direction="minimize")
```

## Custom Strategies

Extend `SearchStrategy` to implement your own algorithm:

```python
from hyperopt import SearchStrategy, SearchSpace, Trial

class MyStrategy(SearchStrategy):
    def __init__(self, search_space: SearchSpace, seed=None):
        super().__init__(search_space, seed)

    def suggest(self) -> dict:
        # Return the next hyperparameter config to evaluate
        return self.search_space.sample()

    def update(self, trial: Trial) -> None:
        # Incorporate trial results (for model-based strategies)
        pass

    @property
    def name(self) -> str:
        return "my_strategy"
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
hyperopt-framework/
├── hyperopt/
│   ├── core/
│   │   ├── search_space.py    # HyperParameter types + SearchSpace
│   │   ├── trial.py           # Trial dataclass + TrialStatus
│   │   ├── strategy.py        # SearchStrategy ABC
│   │   ├── objective.py       # Objective protocol + direction enum
│   │   └── optimizer.py       # Main orchestrator
│   ├── strategies/
│   │   ├── random_search.py   # Random Search
│   │   ├── grid_search.py     # Grid Search
│   │   ├── bayesian.py        # Bayesian Optimization (GP + EI)
│   │   └── hyperband.py       # Hyperband (successive halving)
│   └── tracking/
│       ├── tracker.py         # ExperimentTracker ABC
│       └── sqlite_tracker.py  # SQLite implementation
├── tests/                     # 43 tests
├── examples/
│   └── quickstart.py
└── pyproject.toml
```

## License

MIT
