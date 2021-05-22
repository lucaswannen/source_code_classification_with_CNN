---
title : Classification de langages de programmation à l'aide d'un CNN
---

<font size="8">  __Classification de langages de programmation à l'aide d'un CNN__</font>

---

- [1. Introduction](#1-introduction)
- [2. Demonstration: let's use our classifier !](#2-demonstration--let-s-use-our-classifier--)
- [3. Technical details](#3-technical-details)
  - [3.1. Global architecture and preprocessing](#31-global-architecture-and-préprocessing)
  - [3.2. CNN Architecture](#32-cnn-architecture)
  - [3.3. Model tuning & hyper-parameters](#33-model-tuning--hyper-paramètres)
  - [3.4 Dataset](#34-dataset)
  - [3.5. Results](#35-results)
- [4. Summary of the research article](#4-summary-of-the-research-article)
  - [4.1. Presentation of the article](#41-présentation-of-the-article)
  - [4.2. Goals](#42-goals)
  - [4.3. Model construction](#43-Model-construction)
    - [4.3.1. Entering and Quantizing Characters](#431-Entering-and-quantitizing-characters)
    - [4.3.2. The neural network](#432-the-neural-network)
  - [4.4. Data augmentation](#44-data-augmentation)
  - [5.5. Comparisons and results](#55-comparisons-and-results)

# 1. Introduction

In this project, I have created a programming language classifier that can predict the 4 languages ​​**C**, **Html**, **Python** and **Java**, whose length is greater than or equal to 1024 characters.

I did this with an original approach, given that we relied on a **CNN** with a **character-oriented tokenization**, unlike the standard which is rather to use RNN / LSTM.

This approach is based on the one presented in the research paper This approach is based on the one presented in the research paper [Character-level Convolutional Networks for Text Classification](#41-présentation-of-the-article), without being very faithful to it, and which I have adapted to the classification of programming languages. I wanted a character-based approach, since the tokenizations available and frequent in the literature are based to analyze texts in natural languages, and not those of programming languages.

# 2. Demonstration: let's use our classifier !

You can use our classifier and make predictions on a script (C, html, java or python) longer than 1024 characters:

- Clone the project:

```
$git clone https://github.com/lucaswannen/source_code_classification_with_CNN.git
```

- Launch the python script *predict_a_langage.py* :

```
$python3 predict_a_langage.py your_file
```

# 3. Technical details

I advise you to read the [Summary of the research article](#4-summary-of-the-research-article) 
that I put at the end of this document before reading this part.

## 3.1. Global architecture and preprocessing

As input, there is **a character string** (a java, C, python or html script) of **size greater than or equal to 1024 characters**. This string of characters is cut so as to keep only the first 1024 characters: this allows our CNN to have a fixed size entry.

The characters are then translated into numbers using a derivative of the ASCII table, which processes 101 different characters, including for example tabs or carriage returns, which are important in our case. Then, they are **quantized** with the one-hot encoding method, and this is vectorized and binarized. At this point, each script is therefore encoded in **a 2D** binary matrix of size **1024 *  101**: 1024 being the size of the file and 101 being the number of different possible characters.

## 3.2. CNN Architecture

- __The research paper model__

I first tried to reproduce identically the model proposed in the research article. Their implementation was done on __Torch 7__ but since I did not have access to them, I had to rely solely on their document.

- __The model of our approach__

I made a second model: a CNN performing 2D and not 1D convolutions as in the article, thus allowing the problem to be treated in a way similar to the classification of images in gray levels. This is the originality of my approach.
For this a last preprocessing step is necessary. It artificially increases the dimension of our matrix to bring it to **(1024,101,1)**.

The model is composed of 3 layers of 2D convolutions with 2 layers of Max Pooling interspersed. Following this, we find two dense layers for classification with a dropout between the two.

## 3.3. Model tuning & hyper-parameters

In order to obtain the best performance, we can adjust the hypersparameters of our models or even modify them.

We can in particular play on:

- the probability of dropout
- the number of characters taken into account in each file
- the number of epochs during learning
- the learning rate

## 3.4 Dataset

The dataset I used contains thousands of files from different languages. They represent a total of __150 MB__ of data and __238,000 samples__. The languages ​​are as follows :

- bash
- c
- c#
- c++
- css
- haskell
- html
- java
- javascript
- lua
- markdown
- objective-c
- perl
- php
- python
- r
- ruby
- scala
- sql
- swift
- vb.net

I decided to choose only 4: c, html, java and python, in order to make learning easier for our project. We could very well imagine adding other languages ​​later.
Many of the files in the dataset are smaller than 1024 characters. Thus, one of the stages of preprocessing consists in filtering these files which are too small. Eventually I ended up with around 1500 sample files for each of the 4 languages.

## 3.5. Results

Here are the results for the different models I tested. I find for each model its characteristics, the value of the precision on the test set as well as the confusion matrix obtained. I performed all the calculations on the same test set to keep the comparison consistent.

- ### __Le modèle de l'article__

  - model_article_dense-only-3ep

      Characteristics :

    - dense layer only
    - characters: 1024
    - epoch: 3

      Accuracy: 44.67%

      ![confusion matrix](confusions/model_article_dense-only-3ep.png)

      ---

  - model_article_3ep_1024

      Caractéristiques :

    - caractères : 1024
    - epoch : 3

      Accuracy : 30.52 %

      ![confusion matrix](confusions/model_article_3ep_1024.png)

      ---

  - model_article_3ep_512

      Characteristics :

    - characters: 512
    - epoch: 3

      Accuracy: 24.16%


      ![confusion matrix](confusions/model_article_3ep_512.png)

      ---

  - model_article_3ep_512_2

      Characteristics :

    - characters: 512
    - epoch: 3

      Accuracy: 39.87%

      ![confusion matrix](confusions/model_article_3ep_512_2.png)

      ---

  - model_article_3ep_256_0.001lr

      Characteristics :

    - characters: 256
    - epoch: 3
    - learning rate: 0.001

      Accuracy : 24.16 %

      ![confusion matrix](confusions/model_article_3ep_256_0.001lr.png)

---

- ### __Le modèle de notre approche__

  - model_3_epochs

      Characteristics :

    - characters: 1024
    - epoch : 3

      <!-- Accuracy : 89.84 % -->
      <!-- Accuracy : 88.64 % -->
      Accuracy : 80.56 %

      ![confusion matrix](confusions/model_3_epochs.png)

      ---

  - model_3_epochs_0.5dropout

      Caractéristiques :

    - caractères : 1024
    - epoch : 3
    - dropout : 50 %

      <!-- Accuracy : 88.49 % -->
      <!-- Accuracy : 88.49 % -->
      Accuracy : 80.24 %

      ![confusion matrix](confusions/model_3_epochs_0.5dropout.png)

      ---

  - model_6_epochs_0.5dropout

      Characteristics :

    - characters: 1024
    - epoch : 6
    - dropout : 50 %

      <!-- Accuracy : 87.89 % -->
      <!-- Accuracy : 88.69 % -->
      Accuracy : 84.70 %

      ![matrice de confusion](confusions/model_6_epochs_0.5dropout.png)

      ---

To conlure, the results which we obtain with the last model show an accuracy of 84.70%, which is satisfactory for the project and makes it possible to correctly classify the majority of the codes which one could propose to him.

# 4. Summary of the research article

## 4.1. Présentation de l'article

The research article is titled "__Character-level Convolutional Networks for Text
Classification__ ". It was published on __April 16, 2016__ and written by __Xiang Zhang__, __Junbo Zhao__ and __Yann LeCun__ of the __ Current Institute of Mathematical Sciences__ at __ New York University__.

## 4.2. Goals

This article provides an empirical exploration of the use of convolutional neural networks (CNNs) at the character level for text classification.

The authors indicate that at this time there were already many articles on the use of CNN on words for text classification and some work on the use of characters but still with a learning base on words. . However, this article is the first to discuss CNN based only on characters. This allows, in addition to not needing to understand the syntactic and semantic structure of a language, to prevent the machine from understanding words.
Working only on characters also has an advantage over abnormal character combinations, such as misspellings and emoticons, which can be learned naturally.

## 4.3. Model construction

### 4.3.1. Entering and Quantizing Characters

It all starts with input text having two defined sizes: 1024 and 256. The characters will be __quantified__ with the "one-hot" method. The alphabet is as follows:

`abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}`

It contains 70 characters, so the vectors are of size 70.

As input to the model we therefore have a matrix of either __70 \ * 1024__ or __70 \ * 256__.

### 4.3.2. The neural network

The network is 9 deep with __6 layers of convolution__ then __3 dense__. These are one-dimensional convolutions with temporal __max-pooling__. 2 __dropout__ are placed between the 3 dense layers with a probability of 0.5.

The learning is done thanks to a __stochastic gradient descent__ with __mini-batches__ of size 128. The initial step is 0.01 and is divided by two every 3 epochs (epoch), 10 times.

The weights are initialized with a __Gaussian distribution__ with a mean of 0 and a standard deviation of 0.02 for 1024 characters and 0.05 for 256.

The implementation was done with Torch 7.

## 4.4. Data augmentation

The authors tell us that "data augmentation" is useful in controlling for generalization error. However, for text it is complicated to apply the same methods as for imagery and asking men to rephrase the text is unthinkable with the large amount of data. The choice made here is to replace certain words with their synonyms at random using geometric distributions with one parameter determining the probability of replacing a word and another determining the synonym that will be used.

## 5.5. Comparisons and results

The researchers compared the test errors obtained with this model with traditional methods such as "__Bag of words__", "__Bag of n-grams__" or "__k-means on lexical embedding__" and with deep learning models such as * * CNN ** and a ** LSTM ** based on ** words **.
Of course the tests were carried out under similar conditions to make comparisons.
They also tried to change the alphabet by taking into account the capital letters but that generally brought less good results with the probable cause that it does not change anything in the semantics of the sentence.
They fed these different techniques with 8 datasets of different sizes ranging from hundreds of thousands to several million samples.

The results show that their CNN on characters is the most efficient method on large datasets (from around 1 million samples for training). Below this is the traditional n-gram bag method that looks best.
The authors note that their model seems to make fewer errors than the others on raw data generated by users, which may therefore include language errors. This method would therefore have applications in the real world.

The most important conclusion from these experiments is that character level CNNs can work for text classification without the need for word knowledge. This indicates that language can be thought of as a signal which is no different from any other type of signal.

They take care to specify again that this method does not offer a standard solution to all problems but that a choice must be made according to the situation.
