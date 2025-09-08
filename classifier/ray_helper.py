import jax
import gc

from ray.tune import Stopper, Callback

class GCTuneCallback(Callback):
    def on_trial_complete(self, iteration, trials, trial, **info):
        print(f"GC cleanup after trial: {trial.trial_id}")
        jax.clear_caches()
        jax.clear_backends()
        gc.collect()

class ValidationAccuracyStopper(Stopper):
    def __init__(self, patience=20):
        self.patience = patience
        self.no_improvement_epochs = {}
        self.best_accuracies = {}

    def __call__(self, trial_id, result):
        validation_accuracy = result.get("validation_accuracy", None)

        if validation_accuracy is None:
            return False

        if trial_id not in self.best_accuracies:
            self.best_accuracies[trial_id] = validation_accuracy
            self.no_improvement_epochs[trial_id] = 0
            return False

        epsilon = 1e-6
        if validation_accuracy > self.best_accuracies[trial_id] + epsilon:
            self.best_accuracies[trial_id] = validation_accuracy
            self.no_improvement_epochs[trial_id] = 0
        else:
            self.no_improvement_epochs[trial_id] += 1

        return self.no_improvement_epochs[trial_id] >= self.patience

    def stop_all(self):
        return False
