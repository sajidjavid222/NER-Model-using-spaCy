import spacy
from spacy.tokens import DocBin
import pickle

nlp = spacy.blank("en")


#Load Data
training_data =pickle.load(open(".\Data\TrainData.pickle", "rb"))
testing_data =pickle.load(open(".\Data\TestData.pickle", "rb"))


#training
# the DocBin will store the example documents
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./Data/train.spacy")

#testing
# the DocBin will store the example documents
db_test = DocBin()
for text, annotations in testing_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db_test.add(doc)
db_test.to_disk("./Data/test.spacy")