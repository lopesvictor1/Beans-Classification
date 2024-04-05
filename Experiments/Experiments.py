import pandas as pd
import seaborn as sns
import numpy as np
import random
import pickle
import os
import matplotlib.pyplot as plt
import seaborn.objects as so
from ucimlrepo import fetch_ucirepo
from sklearn.impute import KNNImputer as knni
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



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
        method (str or mlp, optional): the method used to impute the missing values. Defaults to 'knn'.

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
    df['Class'] = df['Class'].transform(lambda x: 0 if x ==' BARBUNYA' else (1 if x == 'BOMBAY' else 
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
    if method == 'knn':
        
        if len(args) == 0:
            n_splits = 10
        else:
            n_splits = args[0]
        
        knn_classifier = KNeighborsClassifier(n_neighbors=10)
        X = df.iloc[:, :16]  # Features (columns 0 to 15)
        y = df.iloc[:, 16]   # Label (column 16)
        scoring = {'acc' : 'accuracy',
                'prec' : 'precision_macro',
                'recall' : 'recall_macro',
                'f1' : 'f1_macro'}
        # Define the cross-validation splitter
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Perform cross-validation and obtain the indices of train and test sets
        cv_results = cross_validate(knn_classifier, X, y, cv=cv, scoring=scoring, return_train_score=True, return_estimator=True, error_score='raise')
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
                                                ('HOROZ' if x == 4 else ('SEKER' if x == 5 else 'SIRA')))))

        # Plot the aggregated confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_aggregated, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=[label_names(i) for i in range(num_classes)],
                    yticklabels=[label_names(i) for i in range(num_classes)])
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.title('Aggregated Confusion Matrix')
        plt.show()

        # Concatenate predictions from all folds
        y_pred = np.concatenate(y_pred)
        print('Average Accuracy: %.2f%%' % (np.mean(cv_results['test_acc']) * 100))
        print('Average Precision: %.2f%%' % (np.mean(cv_results['test_prec']) * 100))
        print('Average Recall: %.2f%%' % (np.mean(cv_results['test_recall']) * 100))
        print('Average F1-Score: %.2f%%' % (np.mean(cv_results['test_f1']) * 100))
        
    elif method == 'mlp':
        pass
    else:
        print("Invalid classification method. Please choose between knn and mlp.")
    return cv_results


            
    

if __name__ == "__main__":
    
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
    print("Introducing missing values...")
    df = introduce_missing_values(df, 5)
    print("Missing values introduced")
    
    #Imputing missing values
    print("Imputing missing values...")
    df = impute_missing_values(df, 'knn')
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
    print("Removing outliers...")
    df = outlier_removal(df, '3sigma')
    print("Outliers removed")
    
    #Normalizing the data
    print("Normalizing the data...")
    df = normalize(df, 'minmax')
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
    cv_results= classificator(df, 'knn', 5)
    print("Data classified")

