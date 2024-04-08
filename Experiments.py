import pandas as pd
import seaborn as sns
import numpy as np
import random
import pickle
import os
import sys
import csv
import matplotlib.pyplot as plt
import seaborn.objects as so
from ucimlrepo import fetch_ucirepo
from sklearn.impute import KNNImputer as knni
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRatio', 'Eccentricity', 
        'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'Roundness', 'Compactness', 'ShapeFactor1', 
        'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']


def introduce_missing_values(df, percentage=5):
    """
    Introduce missing values in the dataframe

    Args:
        df (pd.DataFrame): the original dataframe
        percentage (int, optional): the amount of missing values introduced. Defaults to 5.

    Returns:
        df_imputed (pd.DataFrame): the dataframe with missing values
        """
    for index, i in enumerate(df):
        for jndex, j in enumerate(df[i]):
            if random.randint(0,100) < percentage:
                df.loc[jndex,i] = np.NaN
    return df


def impute_missing_values(df, method='knn'):
    """
    Impute missing values in the dataframe

    Args:
        df (pd.DataFrame): the dataframe with missing values
        method (str or il, optional): the method used to impute the missing values. Defaults to 'knn'.

    Returns:
        df (pd.DataFrame): the dataframe with imputed missing values
    """
    if method == 'knn':
        imputer = knni(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
    elif method == 'il':
        df_imputed = df.interpolate(method="nearest", order=3, limit=None,
                                    limit_direction='both').ffill().bfill()
    else:
        print("Invalid missing value imputer method. Please choose between knn and il.")
    return df_imputed


def add_labels(df, targets):
    """
    Add labels to the dataframe

    Args:
        df (pd.DataFrame): the dataframe
        targets (list): the targets

    Returns:
        df (pd.DataFrame): the dataframe with labels
    """
    df['Class'] = targets
    return df


def transform_labels_int(df):
    """
    Transform the labels in the dataframe int integer numbers

    Args:
        df (pd.DataFrame): the dataframe

    Returns:
        df (pd.DataFrame): the dataframe with transformed labels
    """
    df['Class'] = df['Class'].transform(lambda x: 0 if x == 'BARBUNYA' else (1 if x == 'BOMBAY' else 
                                                ( 2 if x == 'CALI' else (3 if x == 'DERMASON' else 
                                                ( 4 if x == 'HOROZ' else (5 if x == 'SEKER' else (6 if x == 'SIRA' else 99)))))))
    return df


def transform_labels_str(df):
    """
    Transform the labels in the dataframe int string names

    Args:
        df (pd.DataFrame): the dataframe

    Returns:
        df (pd.DataFrame): the dataframe with transformed labels
    """
    df['Class'] = df['Class'].transform(lambda x: 'BARBUNYA' if x == 0 else ('BOMBAY' if x == 1 else
                                                ('CALI' if x == 2 else ('DERMASON' if x == 3 else
                                                ('HOROZ' if x == 4 else ('SEKER' if x == 5 else ('SIRA' if x == 6 else 'UNKNOWN')))))))
    return df


def outlier_removal(df, method='3sigma'):
    """
    Remove the outliers in the dataframe

    Args:
        df (pd.DataFrame): the dataframe
        method (3sigma or MAD, optional): the method used to remove the outliers. Defaults to '3sigma'.

    Returns:
        df (pd.DataFrame): the dataframe without outliers
    """
    df_no_outliers = df.copy()
    
    if method == '3sigma':
        for i in df_no_outliers['Class'].unique():
            class_unique = df_no_outliers[df_no_outliers['Class'] == i]
            for feature in class_unique:
                upper = class_unique[feature].mean() + (3 * class_unique[feature].std())
                lower = class_unique[feature].mean() - (3 * class_unique[feature].std())
                excluded_lower = pd.Series(class_unique[class_unique[feature] < lower].index)
                excluded_upper = pd.Series(class_unique[class_unique[feature] > upper].index)
                df_no_outliers.drop(excluded_lower.values, inplace = False)
                df_no_outliers.drop(excluded_upper.values, inplace = False)
                
    elif method == 'MAD':
        for i in df_no_outliers['Class'].unique():
            class_unique = df_no_outliers[df_no_outliers['Class'] == i]
            for feature in class_unique:
                mad = 1.4826 * np.median(np.absolute(class_unique[feature] - class_unique[feature].median()))
                upper = class_unique[feature].median() + (3 * mad)
                lower = class_unique[feature].median() - (3 * mad)
                excluded_lower = pd.Series(class_unique[class_unique[feature] < lower].index)
                excluded_upper = pd.Series(class_unique[class_unique[feature] > upper].index)
                df_no_outliers.drop(excluded_lower.index, inplace = False)
                df_no_outliers.drop(excluded_upper.index, inplace = False)
    else:
        print("Invalid outlier removal method. Please choose between 3sigma and MAD.")
    
    return df_no_outliers


def normalize(df, method='minmax'):
    """
    Normalize the dataframe

    Args:
        df (pd.DataFrame): the dataframe
        method (minmax or zscore, optional): the method used to normalize the dataframe. Defaults to 'minmax'.

    Returns:
        df (pd.DataFrame): the normalized dataframe
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit_transform(df[cols])
        df_normalized = pd.DataFrame(scaler.transform(df[cols]), columns = cols)
        df_normalized['Class'] = df['Class']
    elif method == 'zscore':
        scaler = StandardScaler()
        scaler.fit_transform(df[cols])
        df_normalized = pd.DataFrame(scaler.transform(df[cols]), columns = cols)
        df_normalized['Class'] = df['Class']
    else:
        print("Invalid normalization method. Please choose between minmax and zscore.")
    return df_normalized


def classificator(df, method='knn', *args):
    """
    Classify the dataframe

    Args:
        df (pd.DataFrame): the dataframe
        method (knn or mlp, optional): the method used to classify the dataframe. Defaults to 'knn'.

    Returns:
        df (pd.DataFrame): the dataframe with the predicted labels
    """
    X = df.iloc[:, :16]  # Features (columns 0 to 15)
    y = df.iloc[:, 16]   # Label (column 16)
    scoring = {'acc' : 'accuracy',
               'prec' : 'precision_macro',
               'recall' : 'recall_macro',
               'f1' : 'f1_macro'}
    
    if method == 'knn':
        
        if len(args) == 0:
            n_neighbors = 10
        elif len(args) == 1:
            n_neighbors = args[0]
        else:
            print("Invalid number of arguments for KNN classification.")
            print("Please provide the number of splits for the cross-validation.")
            print("Example: classificator(df, 'knn', 10)")
            exit(1)
        
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Define the cross-validation splitter
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        
        # Perform cross-validation and obtain the indices of train and test sets
        cv_results = cross_validate(knn_classifier, X, y, cv=cv, scoring=scoring, return_train_score=True, return_estimator=True, error_score='raise')

        agg_conf_matrix, agg_loss = plot_results(cv, cv_results, X, y)
        
    elif method == 'mlp':
        
        if len(args) == 0:
            activation = 'logistic'
            hidden_layer_sizes = (12,3)
            max_iter = 500
            learning_rate = 'constant'
            learning_rate_init = 0.3
            tol = 1e-5
            
            pass
        elif len(args) == 6:
            activation = args[0]
            hidden_layer_sizes = args[1]
            max_iter = args[2]
            learning_rate = args[3]
            learning_rate_init = args[4]
            tol = args[5]
        else:
            print("Invalid number of arguments for MLP classification.")
            print("Please provide the activation function, the hidden layer sizes, the maximum number of iterations," + 
                  "the learning rate, the initial learning rate and the tolerance.")
            print("Example: classificator(df, 'mlp', 'logistic', (12,3), 500, 'constant', 0.3, 1e-3)")
            exit(1)

        classifier = MLPClassifier(activation=activation, solver='adam', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1, 
                                   verbose=True, learning_rate=learning_rate, learning_rate_init=learning_rate_init, tol=tol, max_iter=max_iter)

        # Define the cross-validation splitter
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring, return_train_score=True, return_estimator=True, error_score='raise')

        agg_conf_matrix, agg_loss = plot_results(cv, cv_results, X, y)
  
    else:
        print("Invalid classification method. Please choose between knn and mlp.")
        exit(1)
    return cv_results, agg_conf_matrix, agg_loss


def plot_results(cv, cv_results, X, y):
    """
    Plot the results of the classification

    Args:
        cv (cv): the cross-validation splitter
        cv_results (dict): the results of the classification
        X (pd.DataFrame): the features
        y (pd.DataFrame): the labels
    """
    # Collect predictions from each fold
    y_pred = []
    # Initialize an empty list to store confusion matrices for each fold
    conf_matrices = []
    # Determine the total number of classes
    num_classes = len(np.unique(y))
    for estimator, (_, test_index) in zip(cv_results['estimator'], cv.split(X, y)):
        y_pred_fold = estimator.predict(X.iloc[test_index])
        y_pred.append(y_pred_fold)
        y_true_fold = y.iloc[test_index]
        
        # Compute the confusion matrix for this fold with specified number of classes
        conf_matrix_fold = confusion_matrix(y_true_fold, y_pred_fold, labels=range(num_classes))

        # Append the confusion matrix to the list
        conf_matrices.append(conf_matrix_fold)

    # Aggregate the confusion matrices across all folds
    conf_matrix_aggregated = sum(conf_matrices)

    label_names = lambda x: 'BARBUNYA' if x == 0 else ('BOMBAY' if x == 1 else
                                            ('CALI' if x == 2 else ('DERMASON' if x == 3 else
                                            ('HOROZ' if x == 4 else ('SEKER' if x == 5 else ('SIRA' if x == 6 else 'UNKNOW'))))))

    # Plot the aggregated confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_aggregated, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='d', cbar=False, 
                xticklabels=[label_names(i) for i in range(num_classes)],
                yticklabels=[label_names(i) for i in range(num_classes)])
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    agg_conf_matrix = plt.gcf()

    agg_loss = None
    if isinstance(estimator, MLPClassifier):
        plt.figure(figsize=(12, 8))  # Create the figure outside the loop
        for fold, estimator in enumerate(cv_results['estimator']):
            plt.plot(np.arange(1, estimator.n_iter_ + 1), estimator.loss_curve_, label='Fold {}'.format(fold + 1))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        agg_loss = plt.gcf()
    return agg_conf_matrix, agg_loss
            
    

if __name__ == "__main__":

    missing_data = -1
    missing_data_percentage = -1
    imputing_method = -1
    outlier_method = -1
    normalization_method = -1
    classification_method = -1
    knn_neighbors = -1
    mlp_activation = -1
    mlp_first_layer = -1
    mlp_second_layer = -1
    mlp_max_iter = -1
    mlp_learning_rate = -1
    mlp_learning_rate_init = -1
    mlp_tol = -1
    experiment_name = 'No Experiment Name Specified'

    
    args = sys.argv[1:]

    if len(args) < 1:
        print("Invalid number of arguments. Please provide all arguments necessary.")
        print("Example: python Experiments.py True 5 knn 3sigma minmax knn 10")
        exit(1)
    else:
        if args[0] == 'False':
            missing_data = False
            imputing_method = args[1]
            outlier_method = args[2]
            normalization_method = args[3]
            classification_method = args[4]
            if classification_method == 'knn':
                knn_neighbors = int(args[5])
                experiment_name = args[6]
            else:
                mlp_activation = args[5]
                mlp_first_layer = int(args[6])
                mlp_second_layer = int(args[7])
                mlp_max_iter = int(args[8])
                mlp_learning_rate = args[9]
                mlp_learning_rate_init = float(args[10])
                mlp_tol = float(args[11])
                experiment_name = args[12]
        else:
            missing_data = True
            missing_data_percentage = int(args[1])
            imputing_method = args[2]
            outlier_method = args[3]
            normalization_method = args[4]
            classification_method = args[5]
            if classification_method == 'knn':
                knn_neighbors = int(args[6])
                experiment_name = args[7]
            else:
                mlp_activation = args[6]
                mlp_first_layer = int(args[7])
                mlp_second_layer = int(args[8])
                mlp_max_iter = int(args[9])
                mlp_learning_rate = args[10]
                mlp_learning_rate_init = float(args[11])
                mlp_tol = float(args[12])
                experiment_name = args[13]
    
    #Importing the data
    print("Importing the data...")
    # Check if the files exist in the current directory
    if os.path.exists('df.pkl') and os.path.exists('targets.pkl'):
        # Load the DataFrame and targets from the files
        with open('df.pkl', 'rb') as f:
            df = pickle.load(f)
        with open('targets.pkl', 'rb') as f:
            targets = pickle.load(f)
    else:
        # Fetch the data using fetch_ucirepo
        beans = fetch_ucirepo(id=602)
        df = pd.DataFrame(beans.data.features, columns=beans.feature_names)
        
        # Extract and save the targets
        targets = beans.data.targets
        with open('df.pkl', 'wb') as f:
            pickle.dump(df, f)
        with open('targets.pkl', 'wb') as f:
            pickle.dump(targets, f)
    print("Data imported")
    
    #Introducing missing values
    if missing_data:
        print("Introducing missing values...")
        df = introduce_missing_values(df, missing_data_percentage)
        print("Missing values introduced")
    
        #Imputing missing values
        if imputing_method != "-1":
            print("Imputing missing values...")
            df = impute_missing_values(df, imputing_method)
            print("Missing values imputed")
    

    #Adding labels
    print("Adding labels...")
    df = add_labels(df, targets)
    print("Labels added")
    
    #Transforming labels
    print("Transforming labels into numbers...")
    df = transform_labels_int(df)
    print("Labels transformed into numbers")
    
    #Removing outliers
    if outlier_method != "-1":
        print("Removing outliers...")
        df = outlier_removal(df, outlier_method)
        print("Outliers removed")
    
    #Normalizing the data
    if normalization_method != "-1":
        print("Normalizing the data...")
        df = normalize(df, normalization_method)
        print("Data normalized")

    #Transforming labels into strings
    print("Transforming labels into strings...")
    df = transform_labels_str(df)
    print("Labels transformed into strings")
    
    #Transforming labels
    print("Transforming labels into numbers...")
    df = transform_labels_int(df)
    print("Labels transformed into numbers")
    
    #Classifying the data
    print("Classifying the data...")
    if classification_method == 'knn':
        cv_results, agg_conf_matrix, agg_loss = classificator(df, classification_method, knn_neighbors)
    else:
        cv_results, agg_conf_matrix, agg_loss = classificator(df, classification_method, mlp_activation, (mlp_first_layer, mlp_second_layer), 
                                   mlp_max_iter, mlp_learning_rate, mlp_learning_rate_init, mlp_tol)
    print("Data classified")

    # Save the results to a file
    with open('results.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment_name, missing_data, missing_data_percentage, imputing_method, outlier_method, normalization_method,
                         classification_method, knn_neighbors, mlp_activation, mlp_first_layer, mlp_second_layer, mlp_max_iter, 
                         mlp_learning_rate, mlp_learning_rate_init, mlp_tol, cv_results['test_acc'].mean(), cv_results['test_prec'].mean(), 
                         cv_results['test_recall'].mean(), cv_results['test_f1'].mean()])

    folder_path = os.path.join(os.getcwd(), 'Plots/Best_MLP/')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Save the plots to files
    if agg_conf_matrix != None:
        agg_conf_matrix = plt.title('Aggregated Confusion Matrix - ' + str(experiment_name))
        plt.savefig(folder_path + 'aggregated_confusion_matrix_' + str(experiment_name) + '.png')
    if agg_loss != None:
        agg_loss = plt.title('Training Loss - ' + experiment_name)
        plt.savefig(folder_path + 'training_loss_' + str(experiment_name) + '.png')





