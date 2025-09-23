#!/bin/bash

CONFIG_PATH="conf/conf_classes/gmgm_old.py"
export LC_NUMERIC="en_US.UTF-8"

for sigma in $(seq 1 0.1 2.0)
do
    sigma_name=$(printf "%.1f" "$sigma" | sed 's/\.//')
    exp_name="gmgm_old_sigma_${sigma_name}"

    # Modifie self.sigma
    sed -i "s/self.sigma = .*/self.sigma = $sigma/" "$CONFIG_PATH"
    # Modifie self.experiment_name
    sed -i "s/self.experiment_name = .*/self.experiment_name = \"$exp_name\"/" "$CONFIG_PATH"

    # Nom unique pour la session tmux
    session="gmgm_sigma_${sigma_name}"

    # Supprime la session si elle existe déjà
    tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"

    # Lance la commande dans une nouvelle session tmux détachée
    tmux new-session -d -s "$session" bash -c "python main.py --config gmgm_old"

    sleep 60
done
