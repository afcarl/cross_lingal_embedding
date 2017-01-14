# cross lingal embedding

This script is inspired by [Sebastian Ruder's post](http://sebastianruder.com/cross-lingual-embeddings/) on cross lingal embedding. 

This script learns matrix $W$ s.t. $v\_{en} = Wv\_{ja}$.

# Prerequirements

* gensim
* numpy
* Pandas
* Tensorflow

and

* a parallel translation file like below
* a pair of word2vec model (same dimension size)

example(ja -> en)

```
言語    Language
社会学  Sociology
人工知能    Artificial_intelligence
...
```

I used Wikipedia's `langlink` to generate these pairs.
