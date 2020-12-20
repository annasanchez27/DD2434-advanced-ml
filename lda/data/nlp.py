import spacy


model = None
def nlp(text):
    global model
    if model is None:
        model = spacy.load('en_core_web_md')
    return model(text)
