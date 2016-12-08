import tensorflow as tf
import numpy as np
import pandas as pd
from numpy.linalg import norm

def weight(from_model, from_words, to_model, to_words,
           iteration=100, batch_size=128, dim=200):

    from_vecs = np.array([v/norm(v) for v in [from_model[w] for w in from_words]])
    to_vecs = np.array([v/norm(v) for v in [to_model[w] for w in to_words]])

    from_ = tf.placeholder(tf.float32, shape=[None, dim])
    to_ = tf.placeholder(tf.float32, shape=[None, dim])
    W = tf.Variable(tf.zeros([dim, dim]))

    loss = tf.sub(tf.matmul(from_, W), to_)
    loss_pow = tf.pow(loss, 2)
    train_step = tf.train.AdamOptimizer().minimize(loss_pow)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(iteration):
        for i in range(len(from_vecs//batch_size)):
            batch_f = from_vecs[i * batch_size: (i+1) * batch_size]
            batch_t = to_vecs[i * batch_size: (i+1) * batch_size]
            sess.run(train_step, feed_dict={from_: batch_f, to_: batch_t})
        bar = int(epoch/iteration*50)
        print("\r[{}{}]epoch:{}".format("#"*bar, " "*(50-bar), epoch), end="")

    return W.eval(session=sess)

if __name__ == '__main__':
    from gensim.models import word2vec

    ja_model_path = ""
    en_model_path = ""
    word_pair_path = ""

    ja_model = word2vec.Word2Vec.load_word2vec_format(ja_model_path)
    en_model = word2vec.Word2Vec.load_word2vec_format(en_model_path)

    pair = pd.read_csv(word_pair_path, sep='\t', header=None)

    ja_words, en_words = [], []
    for i in range(len(pair)):
        j, e = pair.iloc[i]
        if (j in ja_model.index2word) and (e in en_model.index2word):
            ja_words.append(j)
            en_words.append(e)

    W = weight(ja_model, ja_words, en_model, en_words)
    np.save("translate_weight.npy", W)
    v = np.dot(ja_model["シンガポール"], W)
    print(en_model.most_similar([v]))


