# tensorflow-XNN

4th Place Solution for [Mercari Price Suggestion Challenge on Kaggle](https://www.kaggle.com/c/mercari-price-suggestion-challenge)

## The Challenge
Build a model to suggest the price of product on Mercari. The model is required to train (including all the preprocessing, feature extraction and model training steps) and inference within 1 hour, using only 4 cores cpu, 16GB RAM, 1GB disk. Data include unstructured text (product title & description) and structured ones, e.g., product category and shipping flag etc.

## Summary
Highlights of our method are as follows:

* very minimum preprocessing with focus on end-to-end learning with multi-field inputs, e.g., textual and categorical;
* hybrid NN consists of four major compoments, i.e., embed, encode, attend and predict. FastText and NN based FM are used as building block;
* purely bagging of NNs of the same architecture via snapshot ensemble;
* efficiency is achieved via various approaches, e.g., lazynadam optimization, fasttext encoding and average pooling, snapshot ensemble, etc.

### Model Architecture
![fig/architecture.png](fig/architecture.png)

Please find the slide of our solution [here](./doc/Mercari_Price_Suggesion_Competition_ChenglongChen_4th_Place.pdf).

## About this project
This is the 4th text mining competition I have attend on Kaggle. The other three are:

* [CrowdFlower Search Results Relevance Competition](https://www.kaggle.