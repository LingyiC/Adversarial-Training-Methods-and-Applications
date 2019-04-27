import numpy as np
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
from nltk import RegexpTokenizer
from keras.utils import to_categorical


def is_number(text):
    try:
        if text[-1] == 's':
            text = text[:-1]
    except:
        pass

    try:
        if text[-1] == 'h' and text[-2] == 't':
            text = text[:-2]
    except:
        pass

    for digits in text:
        if digits not in ['0','1','2','3','4','5','6','7','8','9', '^', '+', '-', '/', ':', '.', '…']:
            return False
    return True


tokenizer = RegexpTokenizer(r'\w+')

dictionary = np.load("../dataset/nltk_dictionary_add2.npy", allow_pickle=True).item()

infile = open("../raw_data/stopwords.txt")
stopwords = [line[:-1] for line in infile.readlines()]
infile.close()

print("Training")
xtrain = list()
# ytrain = list()

train_formal = open('../raw_data/train_formal.csv', 'r')
for line in train_formal:
    if line:
        line_data = line.split('\t')
        if len(line_data) == 3:
            tokens = tokenizer.tokenize(line_data[0].lower())
            tmp_case = []
            for t in tokens:
                if t not in dictionary:
                    ww = dictionary['UNK']
                    tmp_case.append(to_categorical(ww, 2196018))
                elif t in stopwords:
                    continue
                elif is_number(t):
                    ww = dictionary['NUM']
                    tmp_case.append(to_categorical(ww, 2196018))
                else:
                    ww = dictionary[t]
                    tmp_case.append(to_categorical(ww, 2196018))

            xtrain.append(tmp_case)
            # ytrain.append(line_data[1])
        else:
            print(line)
            pass
    else:
        print("empty line: {}.".format(line))
train_formal.close()

unlab = list()

np.save("../dataset/nltk_xtrain_formal_oh.npy", xtrain)
# np.save("../dataset/nltk_ytrain_formal.npy", ytrain)
np.save("../dataset/nltk_ultrain_formal_oh.npy", unlab)

print("Training_mix")
xtrain = list()
# ytrain = list()
unlab = list()
train_mix = open('../raw_data/train_mix.csv', 'r')
for line in train_mix:
    if line:
        line_data = line[:-1].split('\t')
        if len(line_data) == 3:
            tokens = tokenizer.tokenize(line_data[1].lower())
            tmp_case = []
            for t in tokens:
                if t not in dictionary:
                    ww = dictionary['UNK']
                    tmp_case.append(to_categorical(ww, 2196018))
                elif t in stopwords:
                    continue
                elif is_number(t):
                    ww = dictionary['NUM']
                    tmp_case.append(to_categorical(ww, 2196018))
                else:
                    ww = dictionary[t]
                    tmp_case.append(to_categorical(ww, 2196018))

            xtrain.append(tmp_case)
            # ytrain.append(line_data[2])
        else:
            print(line)
            pass
    else:
        print("empty line: {}.".format(line))
train_mix.close()

np.save("../dataset/nltk_xtrain_mix_oh.npy", xtrain)
# np.save("../dataset/nltk_ytrain_mix.npy", ytrain)
np.save("../dataset/nltk_ultrain_mix_oh.npy", unlab)


print("validation")
x_val = list()
y_val = list()

val_mix = open('../raw_data/val_mix.csv', 'r')
for line in val_mix:
    if line:
        line_data = line[:-1].split('\t')
        if len(line_data) == 3:
            tokens = tokenizer.tokenize(line_data[1].lower())
            tmp_case = []
            for t in tokens:
                if t not in dictionary:
                    ww = dictionary['UNK']
                    tmp_case.append(to_categorical(ww, 2196018))
                elif t in stopwords:
                    continue
                elif is_number(t):
                    ww = dictionary['NUM']
                    tmp_case.append(to_categorical(ww, 2196018))
                else:
                    ww = dictionary[t]
                    tmp_case.append(to_categorical(ww, 2196018))

            x_val.append(tmp_case)
            y_val.append(line_data[2])
        else:
            print(line)
            pass
    else:
        print("empty line: {}.".format(line))
val_mix.close()

np.save("../dataset/nltk_xval_oh.npy", x_val)
# np.save("../dataset/nltk_yval.npy", y_val)

print("Test")
xtest = list()
# ytest = list()

test_mix = open('../raw_data/test_mix.csv', 'r')
for line in test_mix:
    if line:
        line_data = line.split('\t')
        if len(line_data) == 3:
            tokens = tokenizer.tokenize(line_data[0].lower())
            tmp_case = []
            for t in tokens:
                if t not in dictionary:
                    ww = dictionary['UNK']
                    tmp_case.append(to_categorical(ww, 2196018))
                elif t in stopwords:
                    continue
                elif is_number(t):
                    ww = dictionary['NUM']
                    tmp_case.append(to_categorical(ww, 2196018))
                else:
                    ww = dictionary[t]
                    tmp_case.append(to_categorical(ww, 2196018))

            xtest.append(tmp_case)
            # ytest.append(line_data[1])
        else:
            print(line)
            pass
    else:
        print("empty line: {}.".format(line))
test_mix.close()

np.save("../dataset/nltk_xtest_oh.npy", xtest)
# np.save("../dataset/nltk_ytest.npy", ytest)
