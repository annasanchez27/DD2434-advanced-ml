from lda.data.document import Document
from lda.data.corpus import Corpus
from experiments.classification import classification
import numpy as np

doc1 = Document.from_text('Mary had a little lamb. Little lamb, little lamb.', topics=['nursery_rhyme', 'lamb'])
doc2 = Document.from_text('Itsy bitsy spider.', topics=['nursery_rhyme'])
doc3 = Document.from_text('Lorem ipsum', topics=['placeholder'])
doc_without_topic = Document.from_text('Doc without topic')
doc_without_topic2 = Document.from_text('Doc with empty topic arr', topics=[])

def create_corpus():
  return Corpus([doc1, doc2, doc3, doc_without_topic, doc_without_topic2])

def create_corpus_with_topics():
  return Corpus([doc1, doc2, doc3])

def test_corpus_with_topics():
  corpus = create_corpus()
  expected_docs_with_topics = create_corpus_with_topics().documents
  
  docs_with_topics = classification.corpus_to_documents_with_topics(corpus)
  
  assert len(docs_with_topics) == len(expected_docs_with_topics)
  assert docs_with_topics == expected_docs_with_topics

def test_topic_to_labels():
  docs_with_topics = create_corpus_with_topics().documents
  labels = classification.topic_to_labels("nursery_rhyme", docs_with_topics)
  assert np.array_equal(np.array([1, 1, 0]), labels)

def test_docs_to_wordcount_dicts():
  expected = [{'mary': 1, 'little': 3, 'lamb': 3}, {'itsy': 1, 'bitsy': 1, 'spider': 1}, {'lorem': 1, 'ipsum': 1}]
  docs_with_topics = create_corpus_with_topics().documents
  wordcount_dicts = classification.docs_to_wordcount_dicts(docs_with_topics)
  assert expected == wordcount_dicts
