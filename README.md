# Xayn Technical Interview task

## About

- The task was to learn model differentiate football articles snippets from other snippets
- I used ready word vectors created by GloVe algorithm
- Each vector represents a word in 300 dimensional space which corresponds to real word meaning
- Combining vectors with GlobalAveragePooling and then applying simple feed forward neural net gave amazing results
- Working on small dataset I was able to achieve almost 90% accuracy over test dataset

## How to run model

- First we need to download pretrained weights for Embeddings layer. You can find them
  there https://nlp.stanford.edu/projects/glove/
- We want to download the 300d version as it is the most accurate and gives words "more" meaning
- Then you should clone the project and install requirements.txt
- After unzipping the GloVe file in project folder you should assign its name in 95th line into the 'path_to_glove_file'
- You are ready to go!

## Data Exploration

Besides, the training and testing part there is also a file in which you can explore data and see basic statistics and
how the data points look like. It's called 'dataAnalysis.py', and it does not require you to download the GloVe word
embeddings.