# Food Review Classification

This project shows a Jupyter Notebook which demonstrates a variety of Recurrent Neural Networks models I built and tuned to classify text for the [Amazong Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) dataset. The dataset has 568,454 food reviews Amazon users left up to October 2012, but for better runtime, only 100,000 samples were selected to be trained in this project. However, it is highly likely that we would be able to achieve higher accuracy scores if more samples were selected, given that your machine is able to handle large-data processing.

## List of different RNN Models:

* Pre-processing: lemmatized the dataset and removed unnecessary punctuations/stop words, then converted text data into sequences and padded these sequences to those of same length.
* MLP: Trained a simple MLP classifier with 3 dense layers and EarlyStopping with patience = 10, which achieved 99% accuracy for train and 70% for test after 11 epochs (out of 100).
* MLP with Dropout: Trained the same MLP classifier but added 1 dropout layer, which only achieved a slight improvement with test accuracy score, 71%.
* SimpleRNN: Trained an RNN classifier with SimpleRNN, which achieved significantly low accuracy scores due to the unstable learning pattern with this dataset.
* LSTM with L1 regularizer: Trained a deeper RNN classifier with 1 LSTM layer with L1 regularizer, 1 dropout layer, and 2 dense layers, which achieved 83% accuracy for train and 71% for test after 13 epochs.
* LSTM with L2 regularizer: Trained the same RNN classifier but with L2 regularizer, which achieved a higher accuracy score for train set, 91%.
* Bidirectional wrapper with LSTM: Trained a deeper RNN classifier with Bidirectional wrapper with 1 LSTM layer, 1 dropout layer, and 2 dense layers, which achieved 91% accuracy for train and 73% for test, the highest among all classifiers.

## Requirements

* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* tensorflow
