from lda import data
from pathlib import Path
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import sparse
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname( __file__ ), '..', 'data', 'classification')


def to_corpus_with_topics(corpus):
  """ Returns list of documents which contain at least one topic. """
  documents_with_topics = []
  for doc in corpus:
    if doc.topics != []:
      documents_with_topics.append(doc)
  return documents_with_topics


def topic_to_labels(label_name, corpus):
  labels = []
  for doc in corpus:
    if label_name in doc.topics:
      labels.append(1)
    else:
      labels.append(0)

  return np.array(labels)


def corpus_to_wordcount_dicts(corpus):
  all_wordcounts = []
  for doc in corpus:
    doc_wordcount = {}
    wordcount = doc.word_count
    for word in doc.included_words:
      doc_wordcount[word.lda_form] = wordcount[word]
    all_wordcounts.append(doc_wordcount)
  
  return all_wordcounts


def corpuses_to_features(X_train_corpus, X_test_corpus, test_size):
  vec = DictVectorizer()
  tt = TfidfTransformer(use_idf=False)

  wordcount_dicts = corpus_to_wordcount_dicts(X_train_corpus)
  wordcounts_matrix = vec.fit_transform(wordcount_dicts).toarray()
  X_train = tt.fit_transform(wordcounts_matrix)
  wordcount_dicts = corpus_to_wordcount_dicts(X_test_corpus)
  wordcounts_matrix = vec.transform(wordcount_dicts).toarray()
  X_test = tt.transform(wordcounts_matrix)

  return X_train, X_test


def classification(label_name, corpus, test_size, random_state=42): 
  label_filename = label_name + '_labels.npy'
  label_file = Path(os.path.join(DATA_DIR, label_filename))
  if not label_file.is_file():
    y = topic_to_labels(label_name, corpus)
    np.save(os.path.join(DATA_DIR, label_filename), y)
  else:
    y = np.load(label_file)

  X_train_corpus, X_test_corpus, y_train, y_test = train_test_split(corpus, y, test_size=test_size, random_state=random_state)
  X_train, X_test = corpuses_to_features(X_train_corpus, X_test_corpus, test_size)
  
  SVM = SVC()
  SVM.fit(X_train, y_train)
  y_pred = SVM.predict(X_test)
  return accuracy_score(y_pred, y_test)*100 


def classification_for_label(corpus, label_name):
  train_data_fractions = [0.01, 0.05, 0.1, 0.2]
  y = []
  yerr = []
  for train_frac in train_data_fractions:
    accuracies = []
    for random_state in range(10):
      acc = classification(label_name, corpus, test_size=1-train_frac, random_state=random_state)
      accuracies.append(acc)

    y.append(np.mean(accuracies))
    yerr.append(np.var(accuracies))
    print(np.mean(accuracies))
    print(np.var(accuracies))
  
  plt.figure()
  plt.errorbar(train_data_fractions, y, yerr=yerr, label='Word features')
  plt.title('Classification accuracy for ' + label_name + ' label')
  plt.xlabel('Proportion of data used for training')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig(os.path.join(DATA_DIR, label_name))


def main():
  reuters = data.reuters
  documents = reuters.documents
  corpus = to_corpus_with_topics(documents)

  classification_for_label(corpus, 'grain')
  classification_for_label(corpus, 'earn')


if __name__ == "__main__":
  main()    