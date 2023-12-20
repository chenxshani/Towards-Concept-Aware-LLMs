import pandas as pd
from nltk.corpus import wordnet as wn
from collections import defaultdict
import spacy
import pickle


df = pd.read_csv("RQ1\\full-triplets-2022Dec16-11720-release.tsv", delimiter="\t")
everyday_things = list(set(df["everyday-thing"]))
et2hypernym_up = defaultdict(list)
et2hyponym_down = defaultdict(list)
nlp = spacy.load('en_core_web_sm')

for et in everyday_things:
    try:
        et2hypernym_up[et] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                              for hyp in wn.synsets(et)[0].hypernyms()]
        et2hyponym_down[et] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                               for hyp in wn.synsets(et)[0].hyponyms()]
    except IndexError:
        try:
            et = et.replace(" ", "_")
            et2hypernym_up[et.replace("_", " ")] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                                                    for hyp in wn.synsets(et)[0].hypernyms()]
            et2hyponym_down[et.replace("_", " ")] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                                                     for hyp in wn.synsets(et)[0].hyponyms()]
        except IndexError:
            try:
                et = et.replace("_", "-")
                et2hypernym_up[et.replace("-", " ")] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                                                        for hyp in wn.synsets(et)[0].hypernyms()]
                et2hyponym_down[et.replace("-", " ")] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                                                         for hyp in wn.synsets(et)[0].hyponyms()]
            except IndexError:
                doc = nlp(et)
                nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
                try:
                    et = et.replace("-", " ")
                    et2hypernym_up[et] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                                          for hyp in wn.synsets(nouns[0])[0].hypernyms()]
                    et2hyponym_down[et] = [hyp._name[:-5].replace("_", " ").replace("-", " ")
                                           for hyp in wn.synsets(nouns[0])[0].hyponyms()]
                except IndexError:
                    et2hypernym_up[et] = []
                    et2hyponym_down[et] = []

with open("100-everyday-things-hypernym-up.pickle", "wb") as pf:
    pickle.dump(et2hypernym_up, pf, protocol=pickle.HIGHEST_PROTOCOL)
with open("100-everyday-things-hyponym-down.pickle", "wb") as pf:
    pickle.dump(et2hyponym_down, pf, protocol=pickle.HIGHEST_PROTOCOL)

print("Done processing and saving the data")
