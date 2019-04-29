from pathlib import Path

import numpy as np


if __name__ == '__main__':
    with Path('vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

    with Path('freqs.words.txt').open() as f:
        word_to_fq = {line.split()[0]: int(line.split()[1]) for _, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    embeddings = np.zeros((size_vocab, 300))

    E_v = np.zeros((1, 300))
    total_count = 0
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path('glove.840B.300d.txt').open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
                E_v += word_to_fq[word] * embeddings[word_idx]
                total_count += word_to_fq[word]
    E_v = E_v / total_count
    print('Got the E(v)')

    Var_v = np.zeros((1, 300))
    print('Reading GloVe file (may take a while)')
    with Path('glove.840B.300d.txt').open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
                Var_v += word_to_fq[word] * (np.linalg.norm(embeddings[word_idx] - E_v))**2
    Var_v = Var_v / total_count
    print('Got the Var(v)')
    print('Normalizing the vectors')
    for i in range(size_vocab):
        embeddings[i, :] = (embeddings[i, :] - E_v) / np.sqrt(Var_v)
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))
    print('Done. Saving...')

    np.savez_compressed('glove_normalized.npz', embeddings=embeddings)
