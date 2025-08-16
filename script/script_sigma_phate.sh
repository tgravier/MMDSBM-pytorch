#!/bin/bash

CONFIG_PATH="conf/conf_classes/phate_dim100_s8.py"
export LC_NUMERIC="en_US.UTF-8"

for sigma in $(seq 1.555 0.025 1.58)
do
    for batch_size in 128 256 512
    do
        sigma_name=$(printf "%.2f" "$sigma" | sed 's/\.//')
        exp_name="phate_dim100_s8_sigma_${sigma_name}_bs${batch_size}"

        # Modifie self.sigma
        sed -i "s/self.sigma = .*/self.sigma = $sigma/" "$CONFIG_PATH"
        # Modifie self.batch_size
        sed -i "s/self.batch_size = .*/self.batch_size = $batch_size/" "$CONFIG_PATH"
        # Modifie self.experiment_name
        sed -i "s/self.experiment_name = .*/self.experiment_name = \"$exp_name\"/" "$CONFIG_PATH"

        # Nom unique pour la session tmux
        session="sigma_${sigma_name}_bs${batch_size}"

        # Supprime la session si elle existe déjà
        tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"

        # Lance la commande dans une nouvelle session tmux détachée
        tmux new-session -d -s "$session" bash -c "python main.py --config phate_dim100_s8"

        sleep 60
    done
done
