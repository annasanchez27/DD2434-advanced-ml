import argparse
import os
from lda import data
from .utils import corpus_to_documents_with_topics
from .classification import plot_classification_for_label


DATA_DIR = os.path.join(os.path.dirname(__file__),
                        '..', 'data', 'classification')


def main():
    parser = argparse.ArgumentParser(
        description='Parse arguments for classification.')
    parser.add_argument('--label', default='grain',
                        help='Classification label', type=str)
    parser.add_argument('--show_fig', action='store_true',
                        default=False, help='Show accuracy plot.')
    parser.add_argument(
        '--seeds', default=5, help='Number of different random seeds for train_test_split.')
    args = parser.parse_args()

    reuters_corpus = data.reuters
    corpus = corpus_to_documents_with_topics(reuters_corpus)

    print(args)
    plot_classification_for_label(
        corpus, args.label, args.show_fig, DATA_DIR, args.seeds)


if __name__ == "__main__":
    main()
