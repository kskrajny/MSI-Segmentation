import optuna
from contrastive_kmeans import contrastive_kmeans
from evaluate import evaluate

if __name__ == '__main__':
    def objective(trial):
        epsilon = trial.suggest_float('epsilon', 0.4, 1.6)
        min_samples = trial.suggest_int('min_samples', 150, 350)
        l = contrastive_kmeans(epsilon, min_samples)
        acc = evaluate()
        return acc * (l > 3) * (l < 33)


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
