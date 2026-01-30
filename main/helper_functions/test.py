# verify SMAC + optuna integration
import logging
import optuna
import optunahub
from optuna.distributions import FloatDistribution

def check_installation():
	print("="*40)
	print(f"optuna version: {optuna.__version__}")

	# 1 Define Search Space explicitly
	search_space = {"x": FloatDistribution(-10, 10)}

	# 2 load SMAC sampler from Optunahub
	print("loading SMAC sampler from Optunahub...", end=" ")
	try:
		smac_module = optunahub.load_module("samplers/smac_sampler", force_reload=False) # will load pre-downloaded module from /urs/local/share/optunahub
		sampler = smac_module.SMACSampler(search_space=search_space) # Instatiate the sampler SMAC
		print("OK")
	except Exception as e:
		print(f"\nFailed to load module: {e}")
		return

	# 2 define opjective
	def objective(trial):
		x = trial.suggest_float("x", -10, 10)
		return (x-2)**2

	# 3 run optimazation
	print("Running 5 test trials with SMAC...", end=" ")
	try:
		study = optuna.create_study(sampler=sampler, direction="minimize")
		optuna.logging.set_verbosity(optuna.logging.WARNING)
		study.optimize(objective, n_trials=5)
		print("OK")
		print(f"Best params: {study.best_params}")
	except Exception as e:
		print(f"\nError during optimization {e}")

if __name__ == "__main__":
	check_installation()


