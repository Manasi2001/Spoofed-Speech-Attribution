"""
Script for decision tree modelling on posterior probabilities for:
    
--BonafideSpoof: bonafide vs spoof classification
--SpoofAttacks: spoof attack algorithm classification

"""

# importing libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# libraries required for visualizing the decision tree
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt

import configparser

config = configparser.ConfigParser()
config.read('config/emb_model_AASIST.conf')


# function for graphing out decision tree
def get_dt_graph(dt_classifier, target_column, df):
    dot_data = StringIO()
    export_graphviz(dt_classifier, out_file=dot_data, filled=True,rounded=True,
                    feature_names=df.iloc[:,1:].columns, 
                    class_names=list(df[target_column].unique()))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph


# function for plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)


# function for evaluating performance on the development set and writing the results in a text file
def dev_data_results(X_dev, y_dev, df, dt_classifier, text_file_path, figure_dir, target_column):
    with open(text_file_path, 'a') as f:
        accuracy = accuracy_score(y_dev, dt_classifier.predict(X_dev))
        f.write(f"Development Set Testing Accuracy: {accuracy}\n")

        f.write("-" * 50 + "\n")

        cnf_matrix = confusion_matrix(y_dev, dt_classifier.predict(X_dev), labels=list(df[target_column].unique()))
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=list(df[target_column].unique()), title='Confusion matrix for Development Data')

        # saving the confusion matrix
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, 'dev_confusion_matrix.png')
        plt.savefig(figure_path)
        plt.close()  

        # writing the classification report in a text file
        report = classification_report(y_dev, dt_classifier.predict(X_dev), target_names=list(df[target_column].unique()))
        f.write(report)
        f.write("-" * 50 + "\n\n")


# function for evaluating performance on the evaluation set and writing the results in a text file
def eval_data_results(X_eval, y_eval, df, df_eval, dt_classifier, text_file_path, figure_dir, task):
    with open(text_file_path, 'a') as f:
        if task == 1:
            accuracy = accuracy_score(y_eval, dt_classifier.predict(X_eval))
            f.write(f"Evaluation Set Testing Accuracy: {accuracy}\n")

            f.write("-" * 50 + "\n")

            cnf_matrix = confusion_matrix(y_eval, dt_classifier.predict(X_eval), labels=list(df['target'].unique()))
            np.set_printoptions(precision=2)
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=list(df['target'].unique()), title='Confusion matrix for Evaluation Data') 

            # writing the classification report in a text file
            report = classification_report(y_eval, dt_classifier.predict(X_eval), target_names=list(df['target'].unique()))
            f.write(report)

            f.write("-" * 50 + "\n\n")

        elif task == 2:
            label_list = list(df['spoof_attack'].unique()) + list(df_eval['spoof_attack'].unique())
            label_list.sort()

            cnf_matrix = confusion_matrix(y_eval, dt_classifier.predict(X_eval), labels=label_list)
            np.set_printoptions(precision=2)
            plt.figure(figsize=(20, 10)) 
            plot_confusion_matrix(cnf_matrix, classes=label_list, title='Confusion matrix for Evaluation Data')
        
        # saving the confusion matrix
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, 'eval_confusion_matrix.png')
        plt.savefig(figure_path)
        plt.close() 


# function to produce results for bonafide versus spoof classification
def bonafide_spoof(df, df_dev, df_eval, save_dir):
    save_dir = os.path.join(save_dir, 'bonafide_spoof_classification')
    os.makedirs(save_dir, exist_ok=True)

    # dropping irrelevant attributes
    df.drop(['ID', 'spoof_attack'], axis=1, inplace = True)
    df_dev.drop(['ID', 'spoof_attack'], axis=1, inplace = True)
    df_eval.drop(['ID', 'spoof_attack'], axis=1, inplace = True)

    # Pearson's correlation matrices
    # for training set
    plt.figure(figsize = (16,8))
    corr = df.drop(['target'], axis = 1).corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="Blues")
    file_path = os.path.join(save_dir, 'train_pearson_corr.png')
    plt.savefig(file_path)
    # for development set
    plt.figure(figsize = (16,8))
    corr = df_dev.drop(['target'], axis = 1).corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="Blues")
    file_path = os.path.join(save_dir, 'dev_pearson_corr.png')
    plt.savefig(file_path)
    # for evaluation set
    plt.figure(figsize = (16,8))
    corr = df_eval.drop(['target'], axis = 1).corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="Blues")
    file_path = os.path.join(save_dir, 'eval_pearson_corr.png')
    plt.savefig(file_path)

    # preparing data for training and testing
    data = df.values
    X, y = data[:,1:], data[:,0]
    data_dev = df_dev.values
    X_dev, y_dev = data_dev[:,1:], data_dev[:,0]
    data_eval = df_eval.values
    X_eval, y_eval = data_eval[:,1:], data_eval[:,0]

    # applying decision tree classifier
    clf = DecisionTreeClassifier(max_depth = int(config['decision_tree']['max_depth']))
    clf = clf.fit(X,y)

    # graphing out decision tree
    gph = get_dt_graph(clf, 'target', df)
    file_path = os.path.join(save_dir, 'decision_tree.png')
    gph.write_png(file_path)

    # path to save the results text file
    text_file_path = os.path.join(save_dir, 'results.txt')

    # ensuring the text file exists (creating it if it doesn't)
    if not os.path.exists(text_file_path):
        open(text_file_path, 'w').close()

    # evaluating results on the development set
    dev_data_results(X_dev, y_dev, df, clf, text_file_path, save_dir, 'target')

    # evaluating results on the evaluation set
    eval_data_results(X_eval, y_eval, df, df_eval, clf, text_file_path, save_dir, 1)


# function to produce results for spoof attack classification
def spoof_attack_classification(df, df_dev, df_eval, save_dir):
    save_dir = os.path.join(save_dir, 'spoof_attack_classification')
    os.makedirs(save_dir, exist_ok=True)

    # dropping all bonafide records
    df = df[df['target'] != 'bonafide']
    df_dev = df_dev[df_dev['target'] != 'bonafide']
    df_eval = df_eval[df_eval['target'] != 'bonafide']

    # dropping irrelevant attributes
    df.drop(['ID', 'target'], axis=1, inplace = True)
    df_dev.drop(['ID', 'target'], axis=1, inplace = True)
    df_eval.drop(['ID', 'target'], axis=1, inplace = True)

    # Pearson's correlation matrices
    # for training set
    plt.figure(figsize = (16,8))
    corr = df.drop(['spoof_attack'],axis = 1).corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="Blues")
    file_path = os.path.join(save_dir, 'train_pearson_corr.png')
    plt.savefig(file_path)
    # for development set
    plt.figure(figsize = (16,8))
    corr = df_dev.drop(['spoof_attack'], axis = 1).corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="Blues")
    file_path = os.path.join(save_dir, 'dev_pearson_corr.png')
    plt.savefig(file_path)
    # for evaluation set
    plt.figure(figsize = (16,8))
    corr = df_eval.drop(['spoof_attack'], axis = 1).corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap="Blues")
    file_path = os.path.join(save_dir, 'eval_pearson_corr.png')
    plt.savefig(file_path)

    # preparing data for training and testing
    data = df.values
    X, y = data[:,1:], data[:,0]
    data_dev = df_dev.values
    X_dev, y_dev = data_dev[:,1:], data_dev[:,0]
    data_eval = df_eval.values
    X_eval, y_eval = data_eval[:,1:], data_eval[:,0]

    # applying decision tree classifier
    clf = DecisionTreeClassifier(max_depth = int(config['decision_tree']['max_depth']))
    clf = clf.fit(X,y)

    # graphing out decision tree
    gph = get_dt_graph(clf, 'spoof_attack', df)
    file_path = os.path.join(save_dir, 'decision_tree.png')
    gph.write_png(file_path)

    # path to save the results text file
    text_file_path = os.path.join(save_dir, 'results.txt')

    # ensuring the text file exists (creating it if it doesn't)
    if not os.path.exists(text_file_path):
        open(text_file_path, 'w').close()

    # evaluating results on the development set
    dev_data_results(X_dev, y_dev, df, clf, text_file_path, save_dir, 'spoof_attack')

    # evaluating results on the evaluation set
    eval_data_results(X_eval, y_eval, df, df_eval, clf, text_file_path, save_dir, 2)


def main():
    # getting the data
    # loading the training set
    df = pd.read_excel(str(config['decision_tree']['train_df']))
    # loading the development set for testing
    df_dev = pd.read_excel(str(config['decision_tree']['dev_df']))
    # loading the evaluation set for testing
    df_eval = pd.read_excel(str(config['decision_tree']['eval_df']))

    # directory to save the decision tree results
    save_dir = 'decision_tree_results'
    # creating the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run specific functions based on provided arguments.")
    parser.add_argument('--BonafideSpoof', action='store_true', help="Run the Bonafide Spoof function")
    parser.add_argument('--SpoofAttacks', action='store_true', help="Run the Spoof Attacks function")

    args = parser.parse_args()

    if args.BonafideSpoof:
        bonafide_spoof(df, df_dev, df_eval, save_dir)
        print('Done!')
    elif args.SpoofAttacks:
        spoof_attack_classification(df, df_dev, df_eval, save_dir)
        print('Done!')
    else:
        print("No function specified. Please use --BonafideSpoof or --SpoofAttacks.")


if __name__ == "__main__":
    main()
