## AUTOMATIC ESSAY SCORING 
- AAYUSH ARORA  102003705
- ANMOL VERMA   102003715

## ABOUT
The Automated Essay Scoring (AES) programme is used to evaluate and score essays prepared in response to certain topics. It is the process of scoring written essays with computer programmes. Because it encourages iterative improvements of students' writings, the process of automating the assessment process could be beneficial to both educators and learners.

## DATASET
The dataset we are using is ‘The Hewlett Foundation: Automated Essay Scoring Dataset’ by ASAP.   https://www.kaggle.com/c/asap-aes/data

## HOW DOES THIS WORK - PROPOSED MODEL

We make a list of words from each sentence and essay. The Word2Vec model is fed this list. By assigning numerical vector values to each word, this model makes meaning of the available words. The features are created by running the essays through the Word2Vec model. In a neural network, the Word2Vec model serves as an Embedding Layer. This model's features are routed through our LSTM layers. We use two LSTM layers. The first layer receives all features from the Embedding Layer (Word2Vec) as input and sends 300 features to the second LSTM layer as output. The second layer accepts 300 features as input and outputs 64 features. We then add a Dropout layer with a value of 0.5. 
## IMPORT LIBRARIES
- FOR ML PART
```S
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```
- FOR NEURAL NETWORK
>first install gensim by using this command if not installed
```s
!pip install gensim==3.8
```
>then proceed towards by importing these libraries
```S
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
```
1. DATA PREPROCESSING
We began by doing some standard preprocessing steps like filling in null values and selecting valid features from the entire dataset after a thorough study.Next we plotted a graph to get a measure of the skewness of our data and applied normalisation techniques to reduce this skewness.The next step involved cleaning the essays to make our training process easier for getting a better accuracy. To achieve this we removed all the unnecessary symbols ,stop words and punctuations from our essays. To increase our accuracy even more we even planned to add some extra features like the number of sentences , number of words,number of characters, average word length etc. Moreover , we even worked on techniques like getting the noun ,verb ,adjective and adverb counts using parts of speech tagging as well as getting the total misspellings in an essay by comparison with a corpus.We applied various machine learning algorithms on this data as explained in the next section.

2. MACHINE LEARNING
In order to utilize machine learning algorithms, it is necessary to convert textual data into a numerical form. This is accomplished through the use of a CountVectorizer, which tokenizes a collection of text documents and produces an encoded vector that represents the vocabulary and the frequency of each word in the document.
Without including the additional features obtained through preprocessing, linear regression, support vector regression (SVR), and random forest algorithms were applied to the dataset. However, the results were not satisfactory, as indicated by a high mean squared error for all three algorithms. Subsequently, the extra features were incorporated into the dataset, and CountVectorizer was applied again. The same three algorithms were then employed, and a marked improvement in performance was observed, particularly for the random forest algorithm, which experienced a significant decrease in mean squared error.

3. NEURAL NETWORK
Preprocessing techniques for neural networks differ from those used for machine learning algorithms. Our training data is inputted into the Word2Vec Embedding Layer, which is a shallow, two-layer neural network that is designed to reconstruct the linguistic contexts of words. This technique generates a vector space, typically consisting of several hundred dimensions, where each unique word in the corpus is assigned a corresponding vector. The vectors are positioned in the space in a way that words with similar contexts in the corpus are situated close to each other. Word2Vec is an efficient predictive model for learning word embeddings from raw text. The features produced by Word2Vec are then passed into the Long Short-Term Memory (LSTM) layer, which has the ability to identify significant data in a sequence that should be retained or discarded. This function significantly aids in the calculation of scores from essays. Finally, the Dense layer with an output of 1 is used to predict the score for each essay.

```S
---------------------THANK YOU-------------------------
~ AAYUSH ARORA AND ANMOL VERMA~
```
