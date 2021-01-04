from collections import Counter
import numpy as np


class np_seed:
    def __init__(self, seed):
        self.seed = seed
    

    def __enter__(self):
        self.old_state = np.random.get_state()
        np.random.seed(self.seed)
    

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.old_state)


def color_by_phis(corpus, phis):
    available_colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    word_topics = {
        document: np.argmax(phis[document], axis=1)
        for document in corpus.documents
    }
    topic_counts = Counter(topic for topics in word_topics.values() for topic in topics)
    sorted_topics = sorted(topic_counts.keys(), key=lambda topic: topic_counts[topic], reverse=True)
    topic_color_mapping = {
        topic: color
        for topic, color in zip(sorted_topics, available_colors)
    }
    return {
        document: document.to_text(colors=[
            (
                topic_color_mapping[topic]
                if (
                    word.include
                    and (topic := word_topics[document][document.included_words.index(word)]) in topic_color_mapping
                )
                else None
            )
            for word in document.words
        ])
        for document in corpus.documents
    }
