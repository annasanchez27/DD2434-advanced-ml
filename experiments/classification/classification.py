from lda import data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction import DictVectorizer


def main():
  reuters = data.reuters
  documents = reuters.documents
  X = []
  y = []
  for doc in documents:
    if doc.topics != []:
      X.append(doc)
      if "grain" in doc.topics:
        y.append(1)
      else:
        y.append(0)
        
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  doc = X_test[1]
  dd = {}
  count = doc.word_count
  for d in doc.included_words:
    print(d)
    dd[d.lda_form] = count[d]
  print(dd)

  # vec = DictVectorizer()
  # print(vec.fit_transform(doc).toarray())

if __name__ == "__main__":
  main()    