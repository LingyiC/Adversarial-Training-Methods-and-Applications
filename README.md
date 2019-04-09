# Milestone-Code-Group-4
Adversarial training (AT) and its variations can be used as means of regularization by adding small perturbations on the training data for supervised and semi-supervised learning. In the domain of NLP, these methods can only serve as regularization methods and are not able to represent real adversarial examples. Given the fact adversarial examples in image classification problems are images with some pixels changed, we hypothesis that informal words(words containing spelling mistakes) may be related to adversarial examples in text classification given its nature of adding or changing some of the character from formal sentences without the change of meaning. Besides, we plan to study whether models trained with AT methods can perform better on language understanding tasks on informal inputs.

This milestone report explains the method and architecture we propose and provide the preliminary results on our experiments for studying the effectiveness of different regularization methods including AT and virtual adversarial training (VAT) over baseline models.Visualizations and analysis showing that the learned word embeddings have improved in quality and the less tendency to overfitting while training. Moreover, in order to find out whether adversarial training can improve the performance of text classification task when input data is informal (or having spelling mistakes), we classified dataset without adversarial training as the first step.

## Experiment 1: Adversarial Text Classification
| Model | Accuracy (with Dropout, p = 0.2) | Accuracy (without Dropout) |
| ------------- | ------------- | ------------- |
| Baseline | 0.906 | 0.909 |
| Adversarial Training | 0.915  | 0.913 |
| Virtual Adversarial Training | 0.919 | 0.918 | 
 
## Experiment 2: Insincere Questions Classification
 
| Embeddings  | F1 Score |
| ------------- | ------------- |
| GoogleNews  | 0.6524  |
| GloVe  | 0.6603  |
| WikiNews | 0.6593 | 
