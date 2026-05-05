# Migration vers MLflow

## Changements effectués

### 1. Dépendances (`pyproject.toml`)
- ✅ Ajout de `mlflow = "^2.18.0"`
- ⚠️ `wandb` est conservé mais optionnel

### 2. Configuration Hydra (`plonk/configs/config.yaml`)

**Avant (WandB):**
```yaml
logger:
  _target_: pytorch_lightning.loggers.WandbLogger
  save_dir: ${root_dir}/plonk
  name: ${experiment_name}${logger_suffix}
  project: diff_plonk
  log_model: False
  offline: False
```

**Après (MLflow):**
```yaml
logger:
  _target_: pytorch_lightning.loggers.MLFlowLogger
  experiment_name: diff_plonk
  tracking_uri: file:${root_dir}/plonk/mlruns
  run_name: ${experiment_name}${logger_suffix}
  log_model: True
  tags:
    mode: ${mode}
    dataset: ${dataset.name}
    model: ${model.name}
```

**Amélioration des checkpoints:**
```yaml
checkpoints:
  save_top_k: 3  # Sauvegarde les 3 meilleurs modèles (au lieu de 0)
  save_on_train_epoch_end: True  # Sauvegarde à la fin de chaque epoch
```

### 3. Scripts d'entraînement

Modifications dans `train.py`, `train_von_fisher.py`, `train_random.py`:

**Supprimé:**
- ❌ Import `wandb`
- ❌ Import `_get_rank` de lightning_fabric
- ❌ Fonction `wandb_init()` et gestion des `wandb_id.txt`
- ❌ Logique de reprise avec `resume="allow"`

**Ajouté:**
- ✅ Logging automatique de la config Hydra complète via `logger.log_hyperparams(dict_config)`
- ✅ Simplification de `load_model()` - plus besoin de gérer les IDs de run

## Utilisation

### Installation des dépendances

```bash
poetry install
```

### Lancer un entraînement

```bash
poetry run python plonk/train.py \
  exp=osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann \
  mode=traineval \
  experiment_name=My_Experiment
```

### Visualiser les résultats

**Option 1: Interface Web MLflow (recommandé)**
```bash
poetry run mlflow ui --backend-store-uri file:./plonk/mlruns
```
Puis ouvrir: http://localhost:5000

**Option 2: MLflow CLI**
```bash
# Lister les expériences
poetry run mlflow experiments list --tracking-uri file:./plonk/mlruns

# Lister les runs d'une expérience
poetry run mlflow runs list --experiment-name diff_plonk

# Voir les détails d'un run
poetry run mlflow runs describe --run-id <run_id>
```

## Fonctionnalités MLflow

### Logs automatiques

MLflow enregistre automatiquement:
- ✅ **Hyperparamètres complets** (toute la config Hydra)
- ✅ **Métriques** (loss, accuracy, etc.) à chaque step/epoch
- ✅ **Checkpoints** (si `log_model: True`)
- ✅ **Tags personnalisés** (mode, dataset, model)
- ✅ **Artifacts** (courbes, images, etc.)
- ✅ **Code source** (version du code)
- ✅ **Environnement** (dépendances)

### Comparaison de runs

Dans l'interface MLflow UI:
1. Sélectionnez plusieurs runs
2. Cliquez sur "Compare"
3. Visualisez les différences de métriques et hyperparamètres

### Requêtes avancées

```python
import mlflow

# Rechercher les meilleurs runs
runs = mlflow.search_runs(
    experiment_names=["diff_plonk"],
    filter_string="metrics.`val/loss` < 0.5",
    order_by=["metrics.`val/loss` ASC"]
)
```

## Structure des dossiers

```
plonk/
├── mlruns/                    # Tracking MLflow (local)
│   └── <experiment_id>/
│       └── <run_id>/
│           ├── artifacts/     # Checkpoints, figures
│           ├── metrics/       # Fichiers de métriques
│           ├── params/        # Hyperparamètres
│           └── tags/          # Tags et metadata
├── checkpoints/               # Checkpoints PyTorch Lightning
│   └── <experiment_name>/
│       ├── last.ckpt
│       └── epoch_*.ckpt
└── datasets/                  # Datasets
```

## Migration depuis WandB (si nécessaire)

Si vous avez des runs WandB existants à migrer:

```python
# Script de migration (à adapter)
import wandb
import mlflow

api = wandb.Api()
runs = api.runs("diff_plonk")

for run in runs:
    with mlflow.start_run(run_name=run.name):
        # Log hyperparams
        mlflow.log_params(run.config)
        
        # Log metrics
        for key, value in run.summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
```

## Avantages de MLflow vs WandB

| Fonctionnalité | WandB | MLflow |
|---------------|-------|--------|
| **Open source** | ❌ | ✅ |
| **Self-hosted** | 💰 | ✅ Gratuit |
| **Offline mode** | ⚠️ Limité | ✅ Natif |
| **Tracking local** | ❌ | ✅ |
| **API Python** | ✅ | ✅ |
| **Model registry** | ✅ | ✅ |
| **Intégration PyTorch Lightning** | ✅ | ✅ |
| **Comparaison de runs** | ✅ | ✅ |
| **Coût** | 💰 Teams | ✅ Gratuit |

## Troubleshooting

### Erreur: "Cannot find experiment"
```bash
# Créer manuellement l'expérience
poetry run mlflow experiments create --experiment-name diff_plonk
```

### Checkpoints non sauvegardés
Vérifier dans la config:
```yaml
logger:
  log_model: True  # Doit être True
checkpoints:
  save_top_k: 3    # Doit être > 0
```

### Métriques non affichées
Les métriques sont loggées via `self.log()` dans les modules PyTorch Lightning.
MLflow les récupère automatiquement.

## Notes

- Les fichiers `wandb_id.txt` ne sont plus utilisés et peuvent être supprimés
- Les anciens logs WandB restent accessibles sur wandb.ai
- Pour garder WandB en parallèle: ajouter un deuxième logger dans la config
  ```yaml
  logger:
    - _target_: pytorch_lightning.loggers.MLFlowLogger
      ...
    - _target_: pytorch_lightning.loggers.WandbLogger
      ...
  ```
