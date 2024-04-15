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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

        #agg_conf_matrix, agg_loss = plot_results(cv, cv_results, X, y)
        plot_loss_curves(cv, cv_results, X, y)
  
    else:
        print("Invalid classification method. Please choose between knn and mlp.")
        exit(1)
    return cv_results, agg_conf_matrix, agg_loss


def mlp(df, activation, hidden_layer_sizes, max_iter, learning_rate, learning_rate_init, tol):
    """
    Classify the dataframe using a MLP

    Args:
        df (pd.DataFrame): the dataframe
        activation (str): the activation function
        hidden_layer_sizes (tuple): the hidden layer sizes
        max_iter (int): the maximum number of iterations
        learning_rate (str): the learning rate
        learning_rate_init (float): the initial learning rate
        tol (float): the tolerance

    Returns:
        df (pd.DataFrame): the dataframe with the predicted labels
    """
    X = df.iloc[:, :16]  # Features (columns 0 to 15)
    y = df.iloc[:, 16]   # Label (column 16)
    scoring = {'acc' : 'accuracy',
            'prec' : 'precision_macro',
            'recall' : 'recall_macro',
            'f1' : 'f1_macro'}
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(activation=activation, solver='adam', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1, 
                                verbose=True, learning_rate=learning_rate, learning_rate_init=learning_rate_init, tol=tol, max_iter=max_iter, 
                                early_stopping=False)

    # Define the cross-validation splitter
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, return_estimator=True, error_score='raise')

    # Train the model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    loss_train = []
    loss_test = []
    
    accuracy_train = []
    accuracy_test = []
    
    for fold, (estimator, (train_index, test_index)) in enumerate(zip(cv_results['estimator'], cv.split(X, y))):
        # Train the estimator
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        
        y_pred = model.predict(X_test)
        
        accuracy_train.append(accuracy_score(y_train, model.predict(X_train)))
        accuracy_test.append(accuracy_score(y_test, y_pred))
        
        
        
        loss_train.append(model.loss_)
        loss_test.append(model.loss_)
        
        print(f'Fold {fold + 1} - Train Accuracy: {accuracy_train[-1]}, Test Accuracy: {accuracy_test[-1]}')
        print(f'Fold {fold + 1} - Train Loss: {loss_train[-1]}, Test Loss: {loss_test[-1]}')
        

    

    # Plot the training loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(loss_train, label='Training Loss')
    plt.plot(loss_test, label='Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_train, label='Training Accuracy')
    plt.plot(accuracy_test, label='Test Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.show()
    
    
def mlpp(df, activation, hidden_layer_sizes, max_iter, learning_rate, learning_rate_init, tol):
    """
    Classify the dataframe using a MLP

    Args:
        df (pd.DataFrame): the dataframe
        activation (str): the activation function ('identity', 'logistic', 'tanh', or 'relu')
        hidden_layer_sizes (tuple): the hidden layer sizes
        max_iter (int): the maximum number of iterations
        learning_rate (str or float): the learning rate ('constant', 'invscaling', or 'adaptive'), or a fixed learning rate
        learning_rate_init (float): the initial learning rate
        tol (float): the tolerance

    Returns:
        df (pd.DataFrame): the dataframe with the predicted labels
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    
    X = df.iloc[:, :16]  # Features (columns 0 to 15)
    y = df.iloc[:, 16]   # Label (column 16)
    
    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    #tol = 1e-20

    # Initialize MLPClassifier
    model = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, 
                          learning_rate=learning_rate, learning_rate_init=learning_rate_init, random_state=42, 
                          tol=tol, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, verbose=True)

    # Train the model
    model.fit(X_train, y_train)
    
    # Get the training and validation losses
    loss_train = model.loss_curve_
    loss_val = model.validation_scores_

    # Plot the training and validation loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(loss_train, label='Training Loss')
    plt.plot(loss_val, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()
    
        # Initialize MLPClassifier
    model = MLPClassifier(activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, 
                          learning_rate=learning_rate, learning_rate_init=learning_rate_init, random_state=42, 
                          tol=tol, early_stopping=False, validation_fraction=0.1, n_iter_no_change=10)

    # Train the model
    model.fit(X_train, y_train)
    
        # Get the training and validation accuracies
    accuracy_train = [accuracy_score(y_train, model.predict(X_train))]
    accuracy_val = [accuracy_score(y_val, model.predict(X_val))]
    epoch = 0
    while epoch < max_iter:
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        
        # Compute accuracy on training and validation sets
        accuracy_train.append(accuracy_score(y_train, model.predict(X_train)))
        accuracy_val.append(accuracy_score(y_val, model.predict(X_val)))
        
        epoch += 1
        # Plot the training and validation accuracy curves
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_train, label='Training Accuracy')
    plt.plot(accuracy_val, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curve')
    plt.legend()
    plt.show()

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
            

def plot_loss_curves(cv, cv_results, X, y):
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
    # Initialize an empty list to store training loss curves for each fold
    training_loss_curves = []
    # Determine the total number of classes
    num_classes = len(np.unique(y))
    for fold, (estimator, (train_index, test_index)) in enumerate(zip(cv_results['estimator'], cv.split(X, y))):
        # Train the estimator
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        # Fit the model
        estimator.fit(X_train, y_train)

        # Predict on test set
        y_pred_fold = estimator.predict(X_test)
        y_pred.append(y_pred_fold)

        # Compute the confusion matrix for this fold with specified number of classes
        conf_matrix_fold = confusion_matrix(y_test, y_pred_fold, labels=range(num_classes))

        # Append the confusion matrix to the list
        conf_matrices.append(conf_matrix_fold)

        # Store the training loss curve for this fold
        training_loss_curves.append(estimator.loss_curve_)
    
    plt.figure(figsize=(12, 8))  # Create the figure outside the loop
    for fold, loss_curve in enumerate(training_loss_curves):
        plt.plot(np.arange(1, len(loss_curve) + 1), loss_curve, label='Fold {}'.format(fold + 1))

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        

if __name__ == "__main__":

    missing_data = True
    missing_data_percentage = 5
    imputing_method = 'knn'
    outlier_method = 'mad'
    normalization_method = 'minmax'
    classification_method = 'mlp'
    knn_neighbors = -1
    mlp_activation = 'logistic'
    mlp_first_layer = 100
    mlp_second_layer = 100
    mlp_max_iter = 500
    mlp_learning_rate = 'constant'
    mlp_learning_rate_init = 0.003
    mlp_tol = 1e-5
    experiment_name = 'No Experiment Name Specified'

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
        mlpp(df, mlp_activation, (mlp_first_layer, mlp_second_layer), 
                                   mlp_max_iter, mlp_learning_rate, mlp_learning_rate_init, mlp_tol)
    print("Data classified")





