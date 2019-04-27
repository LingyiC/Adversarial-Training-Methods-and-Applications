# Introduction 

In this project, we explore adversarial training methods in more depth by performing a series of experiments and also propose a novel model for sequence to sequence classification which has the potential to achieve state of the art results.

We explore adversarial training in 3 phases:

1. IMDB text classification: We tested adversarial training (AT) and virtual adversarial training (VAT) on Large Movie Review Dataset which is a dataset for binary sentiment classification containing both labeled and unlabeled samples to compare the effect of AT and VAT with that of Dropout. In the results, using or not using the Dropout layer seems to not have much effect on the results obtained. But AT and VAT(when unlabeled data is available) significantly improves the results in both cases.

2. Informal questions classification: The experiment can be divided to two sub-parts, one is the comparison of embeddings. We compare three different embeddings, GoogleNews, GloVe and WikiNews, the performance of the different pretrained embeddings are almost similar. The other sub-part is informal and mixed questions classification, in order to find out whether adversarial training can improve the performance of text classification task when input data is informal (or having spelling mistakes). Results show that AT improves performance on informal questions classification. 

3. Sequence to sequence classification: We propose the application of adversarial training to sequence to sequence classification tasks. We take the Bidirectional LSTM with Conditional Random Field (BiLSTM-CRF) model as our baseline model and compare the results with and without adversarial training on the Part-Of-Speech Tagging task. As observed in the results, adversarial training did better than the baseline model.
