# NLP-A4-st124973

## Data

I am using 20 famous classic Romance books from https://www.gutenberg.org/

- Pride and Prejudice by Jane Austen
- Jane Eyre: An Autobiography by Charlotte Brontë
- Wuthering Heights by Emily Brontë
- The Blue Castle: a novel by L. M. Montgomery
- The Enchanted April by Elizabeth Von Arnim
- Sense and Sensibility by Jane Austen
- North and South by Elizabeth Cleghorn Gaskell
- Persuasion by Jane Austen
- The Scarlet Pimpernel by Baroness Emmuska Orczy Orczy
- A Room with a View by E. M. Forster
- The Age of Innocence by Edith Wharton
- Love and Freindship [sic] by Jane Austen
- The Princess and the Goblin by George MacDonald
- The Story of an African Farm by Olive Schreiner
- Lady Susan by Jane Austen
- Emma by Jane Austen
- Our Mutual Friend by Charles Dickens
- This Side of Paradise by F. Scott Fitzgerald
- Anna Karenina by graf Leo Tolstoy
- Middlemarch by George Eliot


## Tokenizer

- NLTK
- '[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[UNK]': 4 (special tokens)


## Training Info

| Model       | Training Loss | Average Cosine Similarity | Accuracy |
|------------------|---------------|--------------|--------------|
| BERT |        5.563886         |       -      |     -     |
| S-BERT |         1.270205          |     0.9921       | 50%        |


- model configs are in config.pkl
- model files download link: https://drive.google.com/file/d/1RgKHpkSc80KBJR8B4j-B4xOiE_2kiOB7/view?usp=sharing, https://drive.google.com/file/d/1L_-TMaTj4x3o2q2mvk6l5E8BpfSvIIk2/view?usp=sharing


## Web App

- Django
- requirements.txt
- Two sentences relation checker

## Discussion

- In pretraining BERT, I think data cleaning is very important because the resources are raw texts. I tried to clean the texts but it can be better. The amount of data is also small for pretraining the BERT.
- In S-BERT, classifier needs more layer or something because it can't learn even the training data is medium size. 