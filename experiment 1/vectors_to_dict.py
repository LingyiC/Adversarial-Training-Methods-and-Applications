import numpy as np

word_embedding = {}
dictionary = {}
with open("./GloVe-1.2/vectors.txt", "r") as f:
    for i, line in enumerate(f):
        sline = line.split(' ')
        try:
            word = sline[0]
            word_embedding[word] = np.asarray(sline[1:], dtype='float32')
            dictionary[word] = i+1
        except:
            print("error at index {}".format(i))

    embedding_mat = np.zeros( shape=(len(dictionary.keys())+1, len(list(word_embedding.values())[0])))
    for word in dictionary.keys():
        idx = dictionary[word]
        embedding_mat[idx] = word_embedding[word]

    np.save("../dataset/nltk_embedding_matrix.npy", embedding_mat)
    np.save("../dataset/nltk_dictionary.npy", dictionary)
    np.save("../dataset/nltk_word_embedding.npy", word_embedding)
