from os import listdir
from os.path import isfile, join
from nltk import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

PATH = "../dataset/train/"

with open("./merged.txt", "w") as merged:

    current_directory = PATH + 'pos'
    files = [f for f in listdir(current_directory) if ( isfile(join(current_directory, f)) and (f[0] != '.') ) ]
    for f in files:
        with open(current_directory+'/'+f, 'r') as t:
            for line in t:
                if line:
                    line = line.replace('<br /><br />', ' ')
                    merged.write(line+'\n')

    current_directory = PATH + 'neg'
    files = [f for f in listdir(current_directory) if ( isfile(join(current_directory, f)) and (f[0] != '.') ) ]
    for f in files:
        with open(current_directory+'/'+f, 'r') as t:
            for line in t:
                if line:
                    line = line.replace('<br /><br />', ' ')
                    merged.write(line+'\n')

    current_directory = PATH + 'unsup'
    files = [f for f in listdir(current_directory) if ( isfile(join(current_directory, f)) and (f[0] != '.') ) ]
    for f in files:
        with open(current_directory+'/'+f, 'r') as t:
            for line in t:
                if line:
                    line = line.replace('<br /><br />', ' ')
                    merged.write(line+'\n')

with open("./merged.txt", "r") as f:
    lines = f.readlines()
    text = ''
    for line in lines:
        text = text + line

# perform TOKENIZATION, returns a vector of words
words = tokenizer.tokenize(text.lower())

with open("./GloVe-1.2/new_merged", "w") as f:
    for w in words:
        f.seek(0, 2)
        f.write(w + ' ')
