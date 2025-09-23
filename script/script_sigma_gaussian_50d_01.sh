#!/bin/bash

CONFIG_PATH="conf/conf_classes/gaussian_50d_01.py"
export LC_NUMERIC="en_US.UTF-8"

for sigma in $(seq 1 0.1 2)
do
    sigma_name=$(printf "%.1f" "$sigma" | sed 's/\.//')
    exp_name="gaussian_50d_01_sigma_${sigma_name}"

    # Modifie self.sigma
    sed -i "s/self.sigma = .*/self.sigma = $sigma/" "$CONFIG_PATH"
    # Modifie self.experiment_name
    sed -i "s/self.experiment_name = .*/self.experiment_name = \"$exp_name\"/" "$CONFIG_PATH"

    # Nom unique pour la session tmux
    session="gauss50d_sigma_${sigma_name}"

    # Supprime la session si elle existe déjà
    tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"

    # Lance la commande dans une nouvelle session tmux détachée
    tmux new-session -d -s "$session" bash -c "python main.py --config gaussian_50d_01"

    sleep 60
done
