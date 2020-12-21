from lda import data
from sklearn.model_selection import train_test_split
import numpy as np

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
        print(doc.topics)
      else:
        y.append(0)
        
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  print(X_test[0].word_count)

if __name__ == "__main__":
  main()    