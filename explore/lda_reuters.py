from lda import data
from lda.data.corpus import Corpus
from lda import utils
from lda import lda
import matplotlib.pyplot as plt

reuters_docs = data.reuters.documents

docs_with_topics = reuters_docs[4:6]

corpus_with_topics = Corpus(docs_with_topics)
with utils.np_seed(123):
    result = lda(corpus_with_topics, num_topics=2, num_iterations=32)

print(result['params'])
plt.figure()
plt.plot(result['lower_bound_evol'])
plt.show()
