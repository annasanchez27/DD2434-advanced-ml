import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from lda import data
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import sparse


def corpus_to_documents_with_topics(corpus):
  """ Returns list of documents which contain at least one topic. """
  all_documents = corpus.documents
  documents_with_topics = []
  for doc in all_documents:
    if doc.topics != [] and doc.topics is not None:
      documents_with_topics.append(doc)
  return documents_with_topics


def topic_to_labels(label_name, documents):
  labels = []
  for doc in documents:
    if label_name in doc.topics:
      labels.append(1)
    else:
      labels.append(0)

  return np.array(labels)


def docs_to_wordcount_dicts(documents):
  all_wordcounts = []
  for doc in documents:
    doc_wordcount = {}
    wordcount = doc.word_count
    for word in doc.included_words:
      doc_wordcount[word.lda_form] = wordcount[word]
    all_wordcounts.append(doc_wordcount)
  
  return all_wordcounts


def docs_to_features(X_train_docs, X_test_docs, test_size):
  vec = DictVectorizer()
  tt = TfidfTransformer(use_idf=False)

  wordcount_dicts = docs_to_wordcount_dicts(X_train_docs)
  wordcounts_matrix = vec.fit_transform(wordcount_dicts).toarray()
  X_train = tt.fit_transform(wordcounts_matrix)
  wordcount_dicts = docs_to_wordcount_dicts(X_test_docs)
  wordcounts_matrix = vec.transform(wordcount_dicts).toarray()
  X_test = tt.transform(wordcounts_matrix)

  return X_train, X_test


def classification(label_name, corpus, dest_dir, test_size=0.8, random_state=42): 
  label_filename = label_name + '_labels.npy'
  label_file = Path(os.path.join(dest_dir, label_filename))
  if not label_file.is_file():
    y = topic_to_labels(label_name, corpus)
    np.save(os.path.join(dest_dir, label_filename), y)
  else:
    y = np.load(label_file)

  X_train_docs, X_test_docs, y_train, y_test = train_test_split(corpus, y, test_size=test_size, random_state=random_state)
  X_train, X_test = docs_to_features(X_train_docs, X_test_docs, test_size)
  
  SVM = SVC()
  SVM.fit(X_train, y_train)
  y_pred = SVM.predict(X_test)
  return accuracy_score(y_pred, y_test)*100 


def classification_for_label(corpus, label_name, show_fig, dest_dir, seed_num):
  train_data_fractions = [0.01, 0.05, 0.1, 0.2]
  y = []
  yerr = []
  for train_frac in train_data_fractions:
    accuracies = []
    for random_state in range(seed_num):
      acc = classification(label_name, corpus, dest_dir, test_size=1-train_frac, random_state=random_state)
      accuracies.append(acc)

    y.append(np.mean(accuracies))
    yerr.append(np.var(accuracies))
    print("Accuracy for training fraction ", train_frac, ":", np.mean(accuracies))
  
  plt.figure()
  plt.errorbar(train_data_fractions, y, yerr=yerr, label='Word features')
  plt.title('Classification accuracy for ' + label_name + ' label')
  plt.xlabel('Proportion of data used for training')
  plt.ylabel('Accuracy')
  plt.legend()
  fig = plt.gcf()
  fig.savefig(os.path.join(dest_dir, label_name))
  if show_fig:
    plt.show()


def main():
  parser = argparse.ArgumentParser(description='Parse arguments for classification.')
  parser.add_argument('--label', default='grain', help='Classification label', type=str)
  parser.add_argument('--show_fig', action='store_true', default=False, help='Show accuracy plot.')
  parser.add_argument('--dest', default='../data/classification', help='Path to directory to save plot and labels.')
  parser.add_argument('--seeds', default=5, help='Number of different random seeds for train_test_split.')
  args = parser.parse_args()

  reuters_corpus = data.reuters
  corpus = corpus_to_documents_with_topics(reuters_corpus)

  print(args)
  classification_for_label(corpus, args.label, args.show_fig, args.dest, args.seeds)

if __name__ == "__main__":
  main()    