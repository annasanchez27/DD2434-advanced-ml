from .utils import docs_to_strings
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.preprocessors import tokenize
from plsa.pipeline import Pipeline
from plsa.corpus import Corpus
from plsa.algorithms.conditional_plsa import ConditionalPLSA
from lda.data import corpus
from lda.data.word import Word
from lda.utils import np_seed
from lda import data
from lda import lda
from lda.variational.e_step import e_step
from lda.variational.lda import corpus_lower_bound
import numpy as np
import collections
import matplotlib.pyplot as plt

def unigram(documents):
    wordcount_dict = {}
    for doc in documents:
        wordcounts = doc.word_count
        for word, count in wordcounts.items():
            if word in wordcount_dict:
                wordcount_dict[word] += count
            else:
                wordcount_dict[word] = count

    model = collections.defaultdict(
        lambda: 1e-10,
        {
            word: count
            for word, count in wordcount_dict.items()
            # if count > 1
        })
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model


def unigram_perplexity(model, documents, total_words):
    perpl_sum = 0
    for doc in documents:
        for word in doc.included_words:
            perpl_sum += np.log(model[word])
    return np.exp(-perpl_sum/total_words)


def compute_topic_sum(word, doc_iter, word_given_topic, topic_given_doc):
    new_word_prob = 1e-10
    topics_num = word_given_topic.shape[0]
    topic_sum = 0
    for topic_iter in range(topics_num):
        if word not in word_given_topic[topic_iter]:
            topic_sum += new_word_prob * \
                topic_given_doc[doc_iter, topic_iter]
        else:
            topic_sum += word_given_topic[topic_iter][word] * \
                topic_given_doc[doc_iter, topic_iter]
    return topic_sum


def plsa_logprobability(train_docs_num, topics_num, doc, word_given_topic, topic_given_doc):
    # log(P(w, 0))
    logp = -np.log(train_docs_num)
    for word in doc:
        a = compute_topic_sum(word, 0, word_given_topic, topic_given_doc)
        logp += np.log(a)

    sum_of_fractions = 0
    for doc_iter in range(1, train_docs_num):
        sum_of_fractions += compute_fraction(doc,
                                             doc_iter, word_given_topic, topic_given_doc)

    logp += np.log(1 + sum_of_fractions)
    return logp


def compute_fraction(doc, doc_iter, word_given_topic, topic_given_doc):
    prod = 1
    for word in doc:
        try:
            numerator = compute_topic_sum(
                word, doc_iter, word_given_topic, topic_given_doc)
            denomintaor = compute_topic_sum(
                word, 0, word_given_topic, topic_given_doc)
            prevprod = prod
            prod *= (numerator / denomintaor)
        except FloatingPointError:
            return 0

    return prod


def plsa_perplexity(train_docs, test_docs, topics_num, total_words, plsa_corpus):
    plsa = ConditionalPLSA(corpus=plsa_corpus, n_topics=topics_num)
    result = plsa.fit(max_iter=10)

    word_given_topic = np.array([
        {
            word_probability[0]: word_probability[1]
            for word_probability in topic_tuple
        }
        for topic_tuple in result.word_given_topic
    ])

    logsum = 0
    docs_num = len(train_docs)
    for i, doc in enumerate(test_docs):
        doc_list = tuple(str(doc).split())
        logsum += plsa_logprobability(docs_num, topics_num, doc_list,
                                      word_given_topic, result.topic_given_doc)

    return np.exp(-logsum/total_words)


def main():
    all_documents = data.reuters.documents
    num_topics = [2, 5, 10, 15, 25]

    chosen_docs = []
    for doc in all_documents:
        if len(doc.included_words) < 100:
            chosen_docs.append(doc)
        if len(chosen_docs) == 800:
            break

    middle = 720
    X_train, X_test = [chosen_docs[:middle], chosen_docs[middle:]]

    total_words = 0
    for doc in X_test:
        total_words += sum(doc.word_count.values())

    print(total_words)

    smoothed_unigram = unigram(X_train)
    uni_perplexity = unigram_perplexity(smoothed_unigram, X_test, total_words)

    # LDA data
    lda_corpus = corpus.Corpus(X_train)
    lda_test_corpus = corpus.Corpus(X_test)

    # pLSI data
    X_train_strings = docs_to_strings(X_train)
    X_test_strings = docs_to_strings(X_test)
    plsa_corpus = Corpus(X_train_strings, Pipeline(tokenize))

    lda_num_iterations = [15, 15, 15, 25, 30]
    plsa_perp_list = []
    lda_perp_list = []
    for i, num_topic in enumerate(num_topics):
        perp = plsa_perplexity(
            X_train_strings, X_test_strings, num_topic, total_words, plsa_corpus)
        
        plsa_perp_list.append(perp)
        with np_seed(123):
            result = lda(lda_corpus, num_topics=num_topic, num_iterations=lda_num_iterations[i])
            alpha = result['params']['alpha']
            beta = result['params']['beta']

            plt.plot(result['lower_bound_evol'])
            plt.show()

            corpus_params = e_step(lda_test_corpus, alpha, beta)
            testset_lower_bound = corpus_lower_bound(lda_test_corpus, alpha, beta, corpus_params['phis'], corpus_params['gammas'])
            lda_perp_list.append(np.exp(testset_lower_bound/total_words))

    uni_perp_list = [uni_perplexity for _ in range(len(num_topics))]
    plt.plot(num_topics, uni_perp_list, label='Smoothed unigram')
    plt.plot(num_topics, plsa_perp_list, label='pLSI')
    plt.plot(num_topics, lda_perp_list, label='LDA')
    plt.legend()
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.show()


if __name__ == "__main__":
    main()
