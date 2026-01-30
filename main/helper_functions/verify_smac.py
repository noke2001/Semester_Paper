import optuna
import optunahub
import smac
import sys

def check_installation():
    print("-" * 40)
    print(f"Optuna Version: {optuna.__version__}")
    print(f"SMAC Version:   {smac.__version__}")
    print("-" * 40)

    # 1. Version Sanity Check
    # SMAC versions 2.0.0+ are incompatible with Optuna's current integration

    module = optunahub.load_module("samplers/smac_sampler")
    SMACSampler = module.SMACSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x**2 + y**2


n_trials = 100
sampler = SMACSampler(
    {
        "x": optuna.distributions.FloatDistribution(-10, 10),
        "y": optuna.distributions.IntDistribution(-10, 10),
    },
    n_trials=n_trials,
    output_directory="smac3_output",
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=n_trials)
print(study.best_trial.params)
version = int(smac.__version__.split('.')[0])
    if major_version >= 2:
        print("❌ CRITICAL ERROR: SMAC version is >= 2.0.0.")
        print("   Optuna integration requires smac < 2.0.0.")
        sys.exit(1)

    # 2. Define a dummy objective
    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
	return x**2 + y**2

    n_trials = 100
    # 3. Attempt to initialize the Sampler
    try:
        print("Attempting to initialize SmacSampler...", end=" ")
        sampler = SmacSampler({
		"x": optuna.distribution.FloatDistribution(-10,10),
		"y": optuna.distribution.FloatDistribution(-10,10),
	},
	n_trials=n_trials,
	output_directory="smac3_output",
	)
        print("✅ OK")
    except ImportError as e:
        print("\n❌ ERROR: Could not import SmacSampler. Is 'optuna-integration' installed?")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize sampler: {e}")
        sys.exit(1)

    # 4. Run a micro-optimization (5 trials) to ensure the loop works
    print("Running 5 test trials...", end=" ")
    try:
        study = optuna.create_study(sampler=sampler, direction="minimize")
        # Suppress optuna logging for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        print("✅ OK")
    except Exception as e:
        print(f"\n❌ ERROR during optimization: {e}")
        sys.exit(1)

    print("-" * 40)
    print("SUCCESS: SMAC Sampler is installed and working correctly.")
    print("-" * 40)

if __name__ == "__main__":
    check_installation()
