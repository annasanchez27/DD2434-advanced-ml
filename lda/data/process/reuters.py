import gzip
import pickle
from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm.auto import tqdm
from ..corpus import Corpus
from ..document import Document
from ..paths import original, processed


def read_files(directory: Path):
    return Corpus.merge([
        read_file(file_path)
        for file_path in tqdm(list(directory.iterdir()))
        if file_path.suffix == '.sgm'
    ])


def read_file(path: Path):
    with open(path, errors='ignore') as file:
        soup = BeautifulSoup(file, features='html.parser')
    return Corpus(
        documents=[
            document
            for article in soup.find_all('reuters')
            if (document := read_article(article)) is not None
            if len(document) > 0
        ]
    )


def read_article(article: Tag):
    try:
        return Document.from_text(
            title=article.title.text,
            topics=[
                topic.text
                for topic in article.topics.children
            ],
            text=article.body.text,
        )
    except AttributeError:
        return None # sometimes articles come without text


reuters = read_files(original / 'reuters')
with gzip.open(processed / 'reuters.pkl.gz', 'wb') as file:
    pickle.dump(reuters, file=file)
