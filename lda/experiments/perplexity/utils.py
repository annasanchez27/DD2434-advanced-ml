from lda.data.document import Document
import numpy as np

def docs_to_strings(docs):
  return np.array([doc_to_string(doc) for doc in docs])

def doc_to_string(doc: Document):
  doc_str = ""
  for word in doc.included_words:
    doc_str += word.lda_form + " "
  return doc_str