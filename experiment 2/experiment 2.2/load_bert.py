from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import numpy as np

config_path = './bert/bert_config.json'
model_path = './bert/bert_model.ckpt'
model = load_trained_model_from_checkpoint(config_path, model_path, training=False)
# model.summary()

count = 0
token_dict = {}
with open('./bert/vocab.txt', 'r') as infile:
    for line in infile.readlines():
        token_dict[line[:-1]] = count
        count += 1

tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
print('Tokens:', tokens)
indices, segments = tokenizer.encode(first='语言模型', max_len=512)

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])
