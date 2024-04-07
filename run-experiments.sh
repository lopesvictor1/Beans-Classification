#!/bin/bash

# Check if the argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide an argument: specific experiment or 'all' for all experiments."
    exit 1
fi

missing_data=("True")
missing_percentage=("5")
imputing_method=("knn" "il")
outlier_method=("3sigma" "mad")
normalization_method=("minmax" "zscore")
classification_method=("knn" "mlp")

# Run specific experiment or all experiments
if [ "$1" == "all" ]; then
    echo "Running all experiments..."

    counter=0

    for im in "${imputing_method[@]}"; do
        for om in "${outlier_method[@]}"; do
            for nm in "${normalization_method[@]}"; do
                for cm in "${classification_method[@]}"; do
                    ((counter++))
                    experiment_name="Experimento$counter"
                    for md in "${missing_data[@]}"; do
                        for mp in "${missing_percentage[@]}"; do
                            for (( i=1; i<=50; i++ )); do
                                if [ "$cm" == "knn" ]; then
                                    echo "Running experiment: $md $mp $im $om $nm $cm 10 $experiment_name"
                                    python Experiments.py "$md" "$mp" "$im" "$om" "$nm" "$cm" "10" "$experiment_name"
                                else
                                    echo "Running experiment: $md $mp $im $om $nm $cm logistic 12 3 500 constant 0.3 0.00001 $experiment_name"
                                    python Experiments.py "$md" "$mp" "$im" "$om" "$nm" "$cm" "logistic" "12" "3" "500" "constant" "0.3" "0.00001" "$experiment_name"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
else
    if [ "$5" == "knn"] && [ $# -lt 6 ]; then
        echo "Please provide the number of neighbors for the KNN algorithm."
        exit 1
    fi
    if [ "$5" == "mlp"] && [ $# -lt 12 ]; then
        echo "Please provide the number of hidden layers, the number of neurons per layer, the activation function, the solver, the learning rate, the maximum number of iterations, and the early stopping parameter for the MLP algorithm."
        exit 1
    fi

    if [ "$1" == "TRUE" ]; then
        if [ "$6" == "knn" ]; then
            echo "Running experiment:" "$1" "$2" "$3" "$4" "$5" "$6" "$7"
            python Experiments.py "$1" "$2" "$3" "$4" "$5" "$6" "$7"
        else
            echo "Running experiment:" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "$14"
            python Experiments.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "$14"
        fi 
        exit 0
    else 
        if [ "$6" == "knn" ]; then
            echo "Running experiment:" "$1" "$2" "$3" "$4" "$5" "$6" 
            python Experiments.py "$1" "$2" "$3" "$4" "$5" "$6"
        else
            echo "Running experiment:" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}"
            python Experiments.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}"
        fi

        exit 0
    fi

fi