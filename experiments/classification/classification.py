import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import topic_to_labels
from utils import docs_to_features
from tqdm.auto import tqdm


def classification_for_label(documents, label_name, dest_dir, test_size=0.8, random_state=42, save_labels=True):
    label_filename = label_name + '_labels.npy'
    label_file = Path(os.path.join(dest_dir, label_filename))
    if not label_file.is_file():
        y = topic_to_labels(label_name, documents)
        if save_labels:
            np.save(os.path.join(dest_dir, label_filename), y)
    else:
        y = np.load(label_file)

    X_train_docs, X_test_docs, y_train, y_test = train_test_split(
        documents, y, test_size=test_size, random_state=random_state)
    X_train, X_test = docs_to_features(X_train_docs, X_test_docs, test_size)

    SVM = SVC()
    SVM.fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    return accuracy_score(y_pred, y_test)*100


DATA_DIR = os.path.join(os.path.dirname(__file__),
                        '..', 'data', 'classification')


def plot_classification_for_label(documents, label_name, seed_num=5):
    train_data_fractions = [0.01, 0.05, 0.1, 0.2]
    y = []
    yerr = []
    for iter, train_frac in enumerate(train_data_fractions):
        accuracies = []
        for random_state in tqdm(range(seed_num)):
            acc = classification_for_label(documents, label_name, DATA_DIR,
                                           test_size=1-train_frac, random_state=random_state)
            accuracies.append(acc)

        y.append(np.mean(accuracies))
        yerr.append(np.var(accuracies))
        print(f'[{iter+1}/{len(train_data_fractions)}] Accuracy for training fraction {train_frac}: {np.mean(accuracies)}')

    plt.figure()
    plt.errorbar(train_data_fractions, y, yerr=yerr, label='Word features')
    plt.title(f'Classification accuracy for {label_name} label')
    plt.xlabel('Proportion of data used for training')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
