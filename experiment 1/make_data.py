import numpy as np
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
from nltk import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

dictionary = np.load("../dataset/nltk_dictionary.npy").item()

pos_files = [f for f in listdir("../dataset/train/pos") if ( isfile(join("../dataset/train/pos", f)) and (f[0] != '.') ) ]
neg_files = [f for f in listdir("../dataset/train/neg") if ( isfile(join("../dataset/train/neg", f)) and (f[0] != '.') ) ]
unl_files = [f for f in listdir("../dataset/train/unsup") if ( isfile(join("../dataset/train/unsup", f)) and (f[0] != '.') ) ]

print( "Labeled..." )
xtrain = list()
ytrain = list()
for p in pos_files:
    pos = open("../dataset/train/pos/"+p, 'r')
    for line in pos:
        if line:
            line = line.replace('<br /><br />', ' ')
            tokens = tokenizer.tokenize(line.lower())
            tokens = [ (dictionary[t] if (t in dictionary) else dictionary['<unk>']) for t in tokens ] 
            
            xtrain.append(tokens)
            ytrain.append(1)     
        else:
            print("empty line: {}.".format(line))
    pos.close()
    
for n in neg_files:
    neg = open("../dataset/train/neg/"+n, 'r')
    for line in neg:
        if line:
            line = line.replace('<br /><br />', ' ')
            tokens = tokenizer.tokenize(line.lower())
            tokens = [ (dictionary[t] if (t in dictionary) else dictionary['<unk>']) for t in tokens ] 
            
            xtrain.append(tokens)
            ytrain.append(0)       
        else:
            print("empty line: {}.".format(line))
    neg.close()

print( "Unlabeled..." )
unlab = list()
for u in unl_files:
    unl = open("../dataset/train/unsup/"+u, 'r')
    for line in unl:
        if line:
            line = line.replace('<br /><br />', ' ')
            tokens = tokenizer.tokenize(line.lower())
            tokens = [ (dictionary[t] if (t in dictionary) else dictionary['<unk>']) for t in tokens ] 
            
            unlab.append(tokens)            
        else:
            print("empty line: {}.".format(line))
    unl.close()
    
xtrain = np.asarray(xtrain)
ytrain = np.asarray(ytrain)
unlab = np.asarray(unlab)

np.save("../dataset/nltk_xtrain.npy", xtrain)
np.save("../dataset/nltk_ytrain.npy", ytrain)
np.save("../dataset/nltk_ultrain.npy", unlab)


pos_files = [f for f in listdir("../dataset/test/pos") if ( isfile(join("../dataset/test/pos", f)) and (f[0] != '.') ) ]
neg_files = [f for f in listdir("../dataset/test/neg") if ( isfile(join("../dataset/test/neg", f)) and (f[0] != '.') ) ]

xtest = list()
ytest = list()
for p in pos_files:
    pos = open("../dataset/test/pos/"+p, 'r')
    for line in pos:
        if line:
            line = line.replace('<br /><br />', ' ')
            tokens = tokenizer.tokenize(line.lower())
            tokens = [ (dictionary[t] if (t in dictionary) else dictionary['<unk>']) for t in tokens ] 
            
            xtest.append(tokens)
            ytest.append(1)
                
        else:
            print("empty line: {}.".format(line))
    pos.close()
    
for n in neg_files:
    neg = open("../dataset/test/neg/"+n, 'r')
    for line in neg:
        if line:
            line = line.replace('<br /><br />', ' ')
            tokens = tokenizer.tokenize(line.lower())
            tokens = [ (dictionary[t] if (t in dictionary) else dictionary['<unk>']) for t in tokens ] 
            
            xtest.append(tokens)
            ytest.append(0)
                
        else:
            print("empty line: {}.".format(line))
    neg.close()
    
xtest = np.asarray(xtest)
ytest = np.asarray(ytest)

np.save("../dataset/nltk_xtest.npy", xtest)
np.save("../dataset/nltk_ytest.npy", ytest)
