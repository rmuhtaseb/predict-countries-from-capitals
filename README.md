# Predict Countries from Capitals
This project aims to use the word embeddings to predict the country for a given city through the use of cosine similarity between country2's embedding and words from words embeddings dictionary. For example, Athens is to Greece as Cairo is to __?

country2's embedding = country1's embedding - city1's embedding + city2's embedding

## Data
You can download the full word embedding dataset `GoogleNews-vectors-negative300.bin.gz` from [Google News Page](https://code.google.com/archive/p/word2vec/).

Or you can load a pickle file which is a subset version (300 dimensions) of the full word embedding dataset extracted under this path `./data/word_embeddings_subset.p`.

## Run
```
python main.py --city1 Athens \
               --country1 Greece \
               --city2 Cairo \
               --data_path ./data/word_embeddings_subset.p
```
