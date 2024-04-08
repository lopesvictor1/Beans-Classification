#!/bin/bash


# Define the lists
activations=('logistic' 'tanh' 'relu')
second_hidden_layer_sizes=(10 50 100)
learning_rates=('constant' 'adaptive')
learning_rate_inits=(0.003 0.03 0.3)
tols=(1e-5)



# Loop over combinations of parameters
for activation in "${activations[@]}"; do
    for (( first_size=8; first_size <= 20; first_size++ )); do
        for (( second_size=3; second_size <=15; second_size++ )); do
            for rate in "${learning_rates[@]}"; do
                for init_rate in "${learning_rate_inits[@]}"; do
                    for tol in "${tols[@]}"; do
                        for (( i=1; i<=5; i++ )); do
                            echo "Activation: $activation, First hidden layer size: $first_size, Second hidden layer size: $second_size, Learning rate: $rate, Learning rate init: $init_rate, Tolerance: $tol"
                            python Experiments.py "True" "5" "knn" "3sigma" "minmax" "mlp" "$activation" "$first_size" "$second_size" "500" "$rate" "$init_rate" "$tol" "Experimento_${activation}_${first_size}_${second_size}_${rate}_${init_rate}_${tol}"
                        done
                    done
                done
            done
        done
    done
done