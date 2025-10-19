import nltk

nltk.download("stopwords")
if nltk.__version__.startswith("3.9"): 
    nltk.download("averaged_perceptron_tagger_eng") 
else: 
    nltk.download("averaged_perceptron_tagger")