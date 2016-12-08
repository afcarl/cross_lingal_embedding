# cross lingal embedding

This script is inspired by [Sebastian Ruder's post](http://sebastianruder.com/cross-lingual-embeddings/) on cross lingal embedding. In the script, a simple Monolingal Linear projection is used, i.e. finding a matrix which transform an original-language word vector into a target-language word vector. E.g. "シンガポール" -> "Singapore"

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
