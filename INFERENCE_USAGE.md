# Usage du mode inférence

Le script principal `main.py` a été modifié pour supporter un mode d'inférence en plus du mode d'entraînement.

## Utilisation

### Mode d'entraînement (existant)
```bash
# Nouvel entraînement
python main.py --config experiment_config_name

# Reprendre un entraînement
python main.py --resume_train --experiment_path experiments/mon_experience
```

### Mode d'inférence (nouveau)
```bash
# Lancer l'inférence sur un modèle entraîné avec un epoch spécifique
python main.py --inference --experiment_path experiments/mon_experience --weight_epoch 25

# Lancer l'inférence avec plusieurs runs pour calculer l'incertitude des métriques
python main.py --inference --experiment_path experiments/mon_experience --weight_epoch 25 --n_runs 10
```

## Arguments

- `--config`: Nom du module de configuration dans `conf/conf_classes/` (requis pour nouvel entraînement)
- `--experiment_path`: Chemin vers le dossier d'expérience existant (requis pour `--resume_train` ou `--inference`)
- `--resume_train`: Flag pour reprendre l'entraînement depuis le dernier checkpoint
- `--inference`: Flag pour lancer l'inférence au lieu de l'entraînement
- `--weight_epoch`: Numéro d'epoch à charger pour l'inférence (requis avec `--inference`, ex: 25 pour 0025_forward.pth)
- `--n_runs`: Nombre de runs d'inférence pour calculer les métriques avec incertitude (défaut: 1)

## Structure du module d'inférence

Le nouveau module `bridge/runners/inference_runner.py` contient la fonction `inference_bridges()` qui :

1. Charge la configuration depuis l'expérience existante
2. Crée une instance du trainer
3. Charge les poids du modèle forward depuis l'epoch spécifié (xxxx_forward.pth)
4. Configure le modèle forward en mode évaluation
5. **Charge les datasets de test sauvegardés** depuis le dossier `datasets_test/` de l'expérience
6. **Effectue n_runs générations** pour calculer les métriques avec incertitude
7. **Calcule SWD et MMD** avec leur moyenne et écart-type
8. Lance l'inférence en utilisant la méthode `inference_test()` en direction forward uniquement

**Note importante**: Le mode inférence charge uniquement le modèle forward car les tests se font forcément en direction forward. Les fichiers de poids sont recherchés dans le dossier `network_weight/` sous le format `xxxx_forward.pth` où `xxxx` est le numéro d'epoch avec padding de zéros (ex: `0025_forward.pth` pour l'epoch 25).

### Métriques avec incertitude

Quand `--n_runs > 1` :
- Chaque run génère de nouveaux échantillons (seed différente à chaque fois)
- Les métriques SWD et MMD sont calculées pour chaque run
- Les résultats finaux affichent : `Métrique: moyenne ± écart-type`
- Les résultats sont également loggés sur Weights & Biases si disponible

### Sauvegarde automatique des datasets de test

Quand `separation_train_test=True` est activé dans la configuration :
- Les datasets de test sont automatiquement sauvegardés dans `{experiment_path}/datasets_test/`
- Chaque dataset est sauvé sous le format `test_dataset_time_{time_value}.pt`
- Le module d'inférence charge automatiquement ces datasets sauvegardés

Le module réutilise l'infrastructure existante pour assurer la compatibilité avec les modèles entraînés.
