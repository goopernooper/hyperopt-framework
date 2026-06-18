from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple, Union

from hyperopt.core.objective import ObjectiveDirection
from hyperopt.core.strategy import SearchStrategy
from hyperopt.core.trial import Trial, TrialStatus
from hyperopt.tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[Dict[str, Any]], Union[float, Tuple[float, Dict[str, Any]]]]


class Optimizer:
    def __init__(
        self,
        strategy: SearchStrategy,
        objective: ObjectiveFn,
        direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.strategy = strategy
        self.objective = objective
        self.direction = direction
        self.tracker = tracker
        self.trials: list[Trial] = []

    def run(self, n_trials: int) -> Trial:
        """Run `n_trials` evaluations and return the best trial."""
        for i in range(n_trials):
            params = self.strategy.suggest()
            trial = Trial(params=params)
            trial.start()

            try:
                result = self.objective(params)
                if isinstance(result, tuple):
                    score, metrics = result
                else:
                    score, metrics = result, {}
                trial.complete(score, metrics)
            except Exception as exc:
                trial.fail(str(exc))
                logger.warning("Trial %s failed: %s", trial.trial_id, exc)

            self.strategy.update(trial)
            self.trials.append(trial)

            if self.tracker:
                self.tracker.log_trial(trial)

            if trial.status == TrialStatus.COMPLETED:
                logger.info(
                    "Trial %d/%d [%s] score=%.6f params=%s",
                    i + 1, n_trials, trial.trial_id, trial.score, trial.params,
                )

        return self.best_trial

    @property
    def best_trial(self) -> Trial:
        completed = [t for t in self.trials if t.status == TrialStatus.COMPLETED]
        if not completed:
            raise RuntimeError("No completed trials")
        if self.direction == ObjectiveDirection.MINIMIZE:
            return min(completed, key=lambda t: t.score)
        return max(completed, key=lambda t: t.score)
