import numpy as np
import matplotlib.pyplot as plt
import os
from lda import utils
from lda import lda
from lda.data.corpus import Corpus
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from .utils import topic_to_labels
from .utils import docs_to_features
from .utils import phis_to_features
from tqdm.auto import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data',)


def classification_for_label(X_train_docs, X_test_docs, y_train, y_test, test_size=0.8):
    X_train, X_test = docs_to_features(X_train_docs, X_test_docs)

    SVM = SVC()
    SVM.fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    return accuracy_score(y_pred, y_test)*100


def lda_classification_for_label(X_train_docs, X_test_docs, y_train, y_test, random_state=42):
    corpus = Corpus(X_train_docs + X_test_docs)
    with utils.np_seed(123):
        result = lda(corpus, num_topics=50, num_iterations=32)

    X_train, X_test = phis_to_features(
        result['params']['phis'], X_train_docs, X_test_docs)

    SVM = SVC()
    SVM.fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    return accuracy_score(y_pred, y_test)*100


def plot_classification_for_label(documents, label_name, seed_num=3, save_labels=True):
    train_data_fractions = [0.05, 0.1, 0.2]
    # train_data_fractions = [0.01, 0.05, 0.1, 0.2]
    y = []
    yerr = []
    y_lda = []
    yerr_lda = []
    for iter, train_frac in enumerate(train_data_fractions):
        accuracies = []
        accuracies_lda = []
        for random_state in tqdm(range(seed_num)):
            label_filename = label_name + '_labels.npy'
            label_file = Path(os.path.join(DATA_DIR, label_filename))
            if save_labels:
                if not label_file.is_file():
                    labels = topic_to_labels(label_name, documents)
                    np.save(os.path.join(DATA_DIR, label_filename), labels)
                else:
                    labels = np.load(label_file)
            else:
                labels = topic_to_labels(label_name, documents)


            X_train_docs, X_test_docs, y_train, y_test = train_test_split(
                documents, labels, train_size=train_frac, random_state=random_state, stratify=labels)
            
            print(y_train)

            acc = classification_for_label(X_train_docs, X_test_docs, y_train, y_test,
                                           test_size=1-train_frac)
            acc_lda = lda_classification_for_label(
                X_train_docs, X_test_docs, y_train, y_test)
            accuracies.append(acc)
            accuracies_lda.append(acc_lda)

        y.append(np.mean(accuracies))
        yerr.append(np.var(accuracies))
        y_lda.append(np.mean(accuracies_lda))
        yerr_lda.append(np.var(accuracies_lda))
        print(f'[{iter+1}/{len(train_data_fractions)}] Accuracy for training fraction {train_frac}: {np.mean(accuracies)}')

    plt.figure()
    plt.errorbar(train_data_fractions, y, yerr=yerr, label='Word features')
    plt.errorbar(train_data_fractions, y_lda, yerr=yerr, label='LDA features')
    plt.title(f'Classification accuracy for {label_name} label')
    plt.xlabel('Proportion of data used for training')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
