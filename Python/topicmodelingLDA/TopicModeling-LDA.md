---
layout: page
title: 'Python: Topic Modeling (LDA)'
permalink: /Python/topicmodelingLDA/
---

# 1. Topic Modeling (LDA)

## 1.1 Downloading NLTK Stopwords & spaCy

NLTK (Natural Language Toolkit) is a package for processing natural languages with Python. To deploy NLTK, NumPy should be installed first. Know that basic packages such as NLTK and NumPy are already installed in Colab.

We are going to use the `Gensim`, `spaCy`, `NumPy`, `pandas`, `re`, `Matplotlib` and `pyLDAvis` packages for topic modeling. The `pyLDAvis` package is not in Colab, so you should manually install it.


```python
pip install --upgrade pyldavis gensim
```

    Collecting pyldavis
    [?25l  Downloading https://files.pythonhosted.org/packages/24/38/6d81eff34c84c9158d3b7c846bff978ac88b0c2665548941946d3d591158/pyLDAvis-3.2.2.tar.gz (1.7MB)
    [K     |████████████████████████████████| 1.7MB 7.6MB/s 
    [?25hCollecting gensim
    [?25l  Downloading https://files.pythonhosted.org/packages/5c/4e/afe2315e08a38967f8a3036bbe7e38b428e9b7a90e823a83d0d49df1adf5/gensim-3.8.3-cp37-cp37m-manylinux1_x86_64.whl (24.2MB)
    [K     |████████████████████████████████| 24.2MB 47.4MB/s 
    [?25hRequirement already satisfied, skipping upgrade: wheel>=0.23.0 in /usr/local/lib/python3.7/dist-packages (from pyldavis) (0.36.2)
    Requirement already satisfied, skipping upgrade: numpy>=1.9.2 in /usr/local/lib/python3.7/dist-packages (from pyldavis) (1.19.5)
    Requirement already satisfied, skipping upgrade: scipy>=0.18.0 in /usr/local/lib/python3.7/dist-packages (from pyldavis) (1.4.1)
    Requirement already satisfied, skipping upgrade: joblib>=0.8.4 in /usr/local/lib/python3.7/dist-packages (from pyldavis) (1.0.1)
    Requirement already satisfied, skipping upgrade: jinja2>=2.7.2 in /usr/local/lib/python3.7/dist-packages (from pyldavis) (2.11.3)
    Requirement already satisfied, skipping upgrade: numexpr in /usr/local/lib/python3.7/dist-packages (from pyldavis) (2.7.3)
    Requirement already satisfied, skipping upgrade: future in /usr/local/lib/python3.7/dist-packages (from pyldavis) (0.16.0)
    Collecting funcy
      Downloading https://files.pythonhosted.org/packages/66/89/479de0afbbfb98d1c4b887936808764627300208bb771fcd823403645a36/funcy-1.15-py2.py3-none-any.whl
    Requirement already satisfied, skipping upgrade: pandas>=0.17.0 in /usr/local/lib/python3.7/dist-packages (from pyldavis) (1.1.5)
    Requirement already satisfied, skipping upgrade: six>=1.5.0 in /usr/local/lib/python3.7/dist-packages (from gensim) (1.15.0)
    Requirement already satisfied, skipping upgrade: smart-open>=1.8.1 in /usr/local/lib/python3.7/dist-packages (from gensim) (4.2.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2>=2.7.2->pyldavis) (1.1.1)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.17.0->pyldavis) (2.8.1)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.17.0->pyldavis) (2018.9)
    Building wheels for collected packages: pyldavis
      Building wheel for pyldavis (setup.py) ... [?25l[?25hdone
      Created wheel for pyldavis: filename=pyLDAvis-3.2.2-py2.py3-none-any.whl size=135593 sha256=39175b0f15f90be2550bdbee9a4dae1fa40963cfb7f69eb488e9c7a8b5657e2b
      Stored in directory: /root/.cache/pip/wheels/74/df/b6/97234c8446a43be05c9a8687ee0db1f1b5ade5f27729187eae
    Successfully built pyldavis
    Installing collected packages: funcy, pyldavis, gensim
      Found existing installation: gensim 3.6.0
        Uninstalling gensim-3.6.0:
          Successfully uninstalled gensim-3.6.0
    Successfully installed funcy-1.15 gensim-3.8.3 pyldavis-3.2.2



```python
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spaCy for Lemmatization
import spacy

# Visualization tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for Gensim (This is optional)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
```

    /usr/local/lib/python3.7/dist-packages/past/types/oldstr.py:5: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
      from collections import Iterable
    /usr/local/lib/python3.7/dist-packages/past/builtins/misc.py:4: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3,and in 3.9 it will stop working
      from collections import Mapping


## 1.2 Adding NLTK Stop words

Download the stopwords from NLTK in order to use them.


```python
import nltk; nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.





    True




```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
```

## 1.3 Importing Dataset

We are going to load a dataset, Charles Dickens's *Our Mutual Friend*, in Colab. Google Drive needs to be mounted in order to load a dataset. If you run this code on your local server, skip the step of mounting Google Drive and then load the dataset with the changed directory.


```python
from google.colab import drive
drive.mount('/content/drive/')
```

    Mounted at /content/drive/


We will load a txt file instead of a csv file. If you wanted to load a csv file or another different type of file, I recommend you use the function `pd.read_csv('path')`.


```python
data = open('drive/My Drive/Colab Notebooks/[your path]/OMF.txt', 'r')
data.columns = ['sentences']
```

Let's look at what's the 20th line.


```python
print(data.readline(20))
```

    In these times of ou


## 2.1 Tokenization and Clean-up

Time to tokenize each sentence into a list of words, getting rid of unnecessary items such as punctuation. If you set `deacc=False`, punctuation marks won't be removed.


```python
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data_words = list(sent_to_words(data))

print(data_words[:1])
```

    [['rs', 'though', 'concerning', 'the', 'exact', 'year', 'there', 'is', 'no']]


## 2.2 Bigram and Trigram

Bigrams and trigrams are words that frequently occur together. For example, on_the_rocks is a trigram. We can implement bigrams and trigrams through the Gensim's `Phrases` function. You might want to change `min_count` and `threshold` later in order to get the best results for your purpose.


```python
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])
```

    ['rs', 'though', 'concerning', 'the', 'exact', 'year', 'there', 'is', 'no']


## 2.3 Functions that Deal with Stopwords, Lemmatization, Bigrams, and Trigrams

Let's create functions to remove stopwords, deal with lemmatization, and make bigrams and trigrams. After that, we will implement them.


```python
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
```


```python
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spaCy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spaCy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Perform lemmatization keeping noun, adjective, verb, and adverb
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])
```

    [['concern', 'exact', 'year']]


## 2.4 Dictionary and Corpus

We are going to create the dictionary using `data_lemmatized` from the previous step.


```python
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
```

    [[(0, 1), (1, 1), (2, 1)]]


Gensim vectorizes each word. The generated corpus shown above is (word_id, word_frequency).

Let's view the word for `word_id=0` from `id2word`.


```python
id2word[0]
```




    'concern'




```python
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
```




    [[('concern', 1), ('exact', 1), ('year', 1)]]



## 3.1 Running the LDA Model

We have everything we need to perform the LDA model. Let's build the LDA model with specific parameters. You might want to change `num_topics` and `passes` later. `passes` is the total number of training iterations, similar to `epochs`.


```python
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
```


```python
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    [(0,
      '0.083*"little" + 0.073*"never" + 0.059*"word" + 0.046*"mean" + '
      '0.042*"woman" + 0.032*"hear" + 0.029*"honour" + 0.026*"yet" + 0.026*"next" '
      '+ 0.021*"point"'),
     (1,
      '0.114*"turn" + 0.068*"eye" + 0.060*"return" + 0.038*"place" + 0.025*"love" '
      '+ 0.019*"bear" + 0.019*"hour" + 0.019*"believe" + 0.018*"change" + '
      '0.018*"lay"'),
     (2,
      '0.183*"say" + 0.040*"would" + 0.037*"man" + 0.035*"hand" + 0.029*"much" + '
      '0.019*"great" + 0.017*"twemlow" + 0.017*"old" + 0.017*"quite" + '
      '0.013*"shake"'),
     (3,
      '0.086*"see" + 0.077*"riderhood" + 0.055*"good" + 0.040*"find" + 0.040*"way" '
      '+ 0.037*"ever" + 0.025*"home" + 0.024*"must" + 0.023*"wife" + '
      '0.022*"feeling"'),
     (4,
      '0.088*"come" + 0.077*"make" + 0.051*"think" + 0.043*"young" + 0.033*"lady" '
      '+ 0.030*"want" + 0.027*"question" + 0.027*"seem" + 0.022*"away" + '
      '0.019*"side"'),
     (5,
      '0.075*"know" + 0.062*"take" + 0.027*"shall" + 0.027*"bella" + 0.026*"dear" '
      '+ 0.025*"day" + 0.024*"give" + 0.024*"put" + 0.022*"podsnap" + 0.020*"sit"'),
     (6,
      '0.096*"may" + 0.088*"look" + 0.057*"could" + 0.038*"head" + 0.037*"cry" + '
      '0.033*"part" + 0.032*"do" + 0.029*"face" + 0.025*"long" + 0.022*"voice"'),
     (7,
      '0.087*"go" + 0.046*"time" + 0.031*"well" + 0.023*"name" + 0.022*"back" + '
      '0.021*"last" + 0.021*"let" + 0.019*"keep" + 0.019*"night" + 0.017*"eugene"'),
     (8,
      '0.066*"tell" + 0.049*"leave" + 0.036*"many" + 0.033*"stand" + '
      '0.027*"sloppy" + 0.025*"suppose" + 0.023*"company" + 0.022*"certain" + '
      '0.020*"throw" + 0.020*"lie"'),
     (9,
      '0.067*"ask" + 0.064*"get" + 0.037*"use" + 0.032*"mind" + 0.032*"still" + '
      '0.032*"bring" + 0.026*"open" + 0.024*"wegg" + 0.022*"alone" + 0.022*"door"')]


## 3.2 Evaluating the LDA Model

After training a model, it is common to evaluate the model. For topic modeling, we can see how good the model is through perplexity and coherence scores.


```python
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```

    
    Perplexity:  -9.15864413363542
    
    Coherence Score:  0.4776129744220124


## 3.3 Visualization
Now we have the test results, so it is time to visualiza them. We are going to visualize the results of the LDA model using the `pyLDAvis` package.


```python
# Visualize the topics
pyLDAvis.enable_notebook()
visualization = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
visualization
```





<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.2.2/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el591406496063327521145522333"></div>
<script type="text/javascript">

var ldavis_el591406496063327521145522333_data = {"mdsDat": {"x": [-0.42675667388857436, 0.040046825171708156, 0.020133026245259276, 0.020257013220089173, 0.05554566876811401, 0.0644395954748757, 0.05841507394667297, 0.04856198573239873, 0.06044989655908434, 0.058907588770371846], "y": [-0.013259594076877122, -0.051231081460915845, 0.3892938442592336, -0.19506340917911227, -0.03019075465702661, -0.02267866241336748, -0.022019748831854452, -0.018488121390485345, -0.018633785510484897, -0.01772868673910964], "topics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "cluster": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Freq": [19.46605948335946, 13.196544829270836, 13.109721883420095, 11.497634134751742, 9.80891653224029, 8.021088059785718, 6.854956432424009, 6.402706014581777, 5.982289655911996, 5.660082974254078]}, "tinfo": {"Term": ["say", "go", "come", "may", "look", "know", "make", "turn", "see", "take", "riderhood", "little", "never", "could", "think", "would", "time", "ask", "man", "get", "good", "eye", "tell", "hand", "young", "word", "return", "much", "head", "cry", "say", "would", "man", "hand", "much", "great", "twemlow", "old", "quite", "shake", "feel", "lightwood", "rather", "better", "veneer", "smile", "clothe", "appear", "care", "air", "child", "fall", "live", "understand", "show", "poor", "learn", "tone", "foot", "action", "go", "time", "well", "name", "back", "last", "let", "keep", "night", "eugene", "work", "pay", "room", "repeat", "light", "always", "answer", "wave", "try", "draw", "master", "resume", "pretty", "present", "stop", "dress", "rise", "case", "also", "pass", "know", "take", "shall", "bella", "dear", "day", "give", "put", "podsnap", "sit", "hold", "laugh", "even", "set", "thing", "first", "right", "river", "bad", "call", "glance", "can", "retort", "hope", "half", "rest", "person", "indeed", "heart", "add", "come", "make", "think", "young", "lady", "want", "question", "seem", "away", "side", "friend", "life", "begin", "opinion", "enough", "short", "manner", "pursue", "strike", "sense", "moment", "wait", "boat", "allow", "wear", "thank", "less", "cross", "guest", "sometimes", "may", "look", "could", "head", "cry", "part", "do", "face", "long", "voice", "arm", "fire", "speak", "else", "table", "follow", "round", "tippin", "state", "walk", "member", "watch", "almost", "hat", "fellow", "knowledge", "relation", "lean", "lip", "kiss", "see", "riderhood", "good", "find", "way", "ever", "home", "must", "wife", "feeling", "money", "society", "reply", "husband", "boot", "water", "flush", "miss", "move", "world", "observe", "hair", "touch", "buy", "direction", "break", "brewer", "perhaps", "subject", "blow", "ask", "get", "use", "mind", "still", "bring", "open", "wegg", "alone", "door", "occasion", "corner", "however", "family", "paper", "venus", "marry", "pound", "dinner", "appearance", "carriage", "hard", "school", "far", "possible", "business", "sleep", "offer", "bar", "drop", "little", "never", "word", "mean", "woman", "hear", "honour", "yet", "next", "point", "close", "comfortable", "small", "window", "notice", "really", "stare", "catch", "other", "story", "slight", "attention", "save", "ease", "scream", "remark", "sum", "mother", "explain", "approach", "tell", "leave", "many", "stand", "sloppy", "suppose", "company", "certain", "throw", "lie", "sir", "house", "veneering", "meet", "sure", "beg", "talk", "noble", "kind", "event", "piece", "true", "sort", "write", "low", "marriage", "accept", "spirit", "hide", "strong", "turn", "eye", "return", "place", "love", "bear", "hour", "believe", "change", "lay", "force", "soon", "induce", "likewise", "wish", "power", "confess", "creature", "grateful", "happy", "lose", "course", "blood", "tear", "bright", "visit", "order", "silence", "ring", "enjoyment"], "Freq": [4063.0, 1305.0, 1156.0, 1072.0, 990.0, 1130.0, 1015.0, 735.0, 784.0, 932.0, 705.0, 604.0, 538.0, 635.0, 670.0, 878.0, 699.0, 523.0, 830.0, 500.0, 503.0, 441.0, 448.0, 767.0, 567.0, 430.0, 385.0, 655.0, 424.0, 416.0, 4062.1705098223742, 877.8369341435633, 829.6252220251016, 766.6362818471968, 654.9969011083771, 428.75363970992555, 378.1256933878804, 374.09942263359704, 370.37945898467467, 284.5546247519194, 240.00752830164237, 222.48866129354607, 222.37150979188746, 220.4991134670739, 199.30878212476517, 156.09644102529106, 151.10105299183317, 144.8300233200836, 144.5818264217887, 144.5122224564797, 141.71174912157318, 139.45930355930548, 139.20707194839602, 138.37089327716157, 136.1984695150819, 133.2659162418431, 130.4599081088628, 128.44476572083082, 126.75947660716153, 120.3843410045403, 1304.5256525288687, 698.8693393644297, 468.9100849886702, 348.9127866611214, 324.8141796014737, 318.3344757474148, 312.39595586152836, 288.0701699218713, 280.5220127677336, 250.6266131254605, 235.84567853675364, 222.83863564238538, 216.56312065942407, 214.70324436799623, 206.46983345829554, 202.32498312676336, 200.92569251723157, 173.37945378558163, 171.45792948367952, 170.4871170939333, 169.07113453162236, 159.5132382091724, 151.38797432125935, 138.40594130235766, 138.36787242854035, 135.9944310387065, 130.487264877076, 128.76715212975867, 127.52513057447888, 127.45592209127523, 1129.2459179077964, 931.4071392501869, 407.5646333822423, 399.2685226784556, 390.34817131552677, 370.71194410435464, 356.73354059848634, 353.1044781335467, 336.19413059873074, 298.2940915190116, 277.78935071787976, 269.78096091971315, 252.9750390177293, 247.7725690393653, 235.70136162563233, 230.9141974333542, 228.89304446567834, 212.64152318123612, 194.84755698760668, 190.67720633113217, 179.24606924279405, 175.77584688842995, 160.8945950235469, 160.70537756047952, 150.96536719375442, 144.5667401900449, 142.79089131102344, 142.3188718439092, 141.5665446599481, 137.47665218388093, 1155.9436200471496, 1014.1304866247763, 669.3482402907159, 566.8754410465989, 430.7996983707277, 396.1431784305513, 360.3721722810135, 356.7488272221684, 285.97177917281056, 250.54621315303797, 237.5628899937933, 235.73630778637198, 230.95369859997894, 200.1668267107685, 164.1913990990293, 162.49329676439586, 156.81735344492495, 155.94921911187453, 151.6856744776473, 137.42863708064272, 136.57603569901562, 128.12163722132897, 100.53692420854557, 93.73965089309083, 89.23936012591899, 87.79431291217224, 84.99183728501667, 84.10873216872977, 83.45363513639639, 77.36955771328012, 1072.059846462784, 989.3509339929326, 634.9197340868451, 423.9385701626899, 415.16181393111066, 373.96362375970415, 361.7540603526645, 326.0147563259968, 283.44726756997454, 250.0962027729536, 246.4881749233871, 162.75383678772099, 154.8682079056909, 153.6683435216522, 150.25038308187712, 143.45542626448113, 136.60847459395134, 123.24915453581086, 121.03246600066262, 120.25361708973348, 103.08306491341246, 98.27838391196393, 92.86755156271217, 86.6604900297873, 83.20131008893651, 79.1969088026314, 74.10722573424084, 73.13865868714772, 68.89104305625078, 68.29288329756464, 783.1614059399825, 704.6083037120196, 502.8537720878292, 369.0257299421492, 364.6142275752154, 341.4972362801841, 232.67181128083573, 221.47032341427098, 208.36163951715852, 198.94236000218223, 190.46836588399512, 189.32141209988652, 184.02810716409877, 129.80653373499675, 112.03696039064778, 99.39534334322327, 99.17869519110636, 90.18155175718805, 88.82067806497804, 85.31834596467846, 81.36846248516417, 80.77411452639144, 80.06189790038735, 79.27797961135137, 77.15446793342615, 72.3352550756662, 70.41324553269068, 68.12719445466185, 67.7095599762294, 66.60081439698139, 522.6722631409109, 499.8699424993288, 289.3743293001561, 252.06722229345948, 250.86405679879095, 247.51511696229124, 206.47090191889285, 184.84631772556435, 174.08768893172117, 169.15481824022402, 126.75081226326488, 122.23399428790957, 121.92086727654032, 116.93667945785124, 111.37379503360128, 110.98069403557028, 104.99710097521887, 104.12691026339066, 103.7473908534735, 100.01321189317821, 95.86340281980472, 92.78498723410469, 88.20925256267185, 80.40961755775517, 77.95905168567181, 75.01754492006783, 63.3440679285865, 62.87281450882319, 62.47255840374939, 60.28344903224702, 603.2470098966753, 537.0725755659333, 429.19179830961394, 336.7353745480556, 309.8987324335442, 232.85471161321834, 212.22303372397005, 189.53168201742648, 188.65590244375804, 152.28960583459963, 128.3004349037537, 120.14845600875228, 114.45995169588491, 113.55567046038438, 113.37166868604005, 108.92458596815257, 107.4632564938078, 84.337424583185, 74.67853261045981, 74.65190928164245, 72.06893596977832, 63.91493582511975, 61.04249506235643, 59.568595729032715, 59.51905809854663, 59.39853173010198, 53.71677172164616, 53.654888815476355, 51.14306076932202, 49.947409930467174, 447.2724215649002, 335.942408918226, 249.09646620577732, 228.393550805042, 183.034486707014, 171.20549927117452, 159.4909045981603, 150.9469415720377, 138.55358259326758, 137.08674816780533, 121.65444593114371, 118.79274468942747, 114.07521004178201, 112.4434567337941, 111.54920927369551, 105.87338883830205, 98.41855428367711, 95.76005757345659, 89.68337284934536, 83.86830990587248, 83.3699733180247, 79.10843276752922, 75.20515330892384, 74.07733223254614, 70.47313993304309, 69.76756740671618, 69.03050666685338, 68.59375395382288, 68.17428414959501, 67.50147793270884, 734.5281532226779, 440.1857715833253, 384.37101587316823, 247.34016545150126, 159.0679868182423, 122.12538556737468, 121.68452997540615, 119.53505965827797, 117.45387328679301, 115.23318311516931, 114.01325282496177, 113.08774365650642, 92.67925456610483, 89.93305475853501, 88.08963489064169, 86.03794598043325, 82.24154510507759, 78.96006534193833, 75.12433050740951, 72.38276317897143, 72.36979223740256, 72.13948583007502, 67.3245953122626, 66.33835855373478, 56.51210024085746, 54.20981225824179, 50.798755435960395, 50.15784893998059, 47.10196188624717, 46.549905738393505], "Total": [4063.0, 1305.0, 1156.0, 1072.0, 990.0, 1130.0, 1015.0, 735.0, 784.0, 932.0, 705.0, 604.0, 538.0, 635.0, 670.0, 878.0, 699.0, 523.0, 830.0, 500.0, 503.0, 441.0, 448.0, 767.0, 567.0, 430.0, 385.0, 655.0, 424.0, 416.0, 4063.0877456815997, 878.7541375546364, 830.5424780500434, 767.5535019659695, 655.9141548192987, 429.6710135692103, 379.0442912337295, 375.0165930397281, 371.2967103581455, 285.4718430759455, 240.9247733099243, 223.40600369469183, 223.2887625662942, 221.41632592024786, 200.23582978515788, 157.01367306253505, 152.0186384293261, 145.7472432395113, 145.49919799949384, 145.4294050946648, 142.6289219155954, 140.37651905986084, 140.12431559562452, 139.2881891461129, 137.11568812395507, 134.1830646580114, 131.37746326279554, 129.3620238721696, 127.67668334568496, 121.30174741017439, 1305.438660642263, 699.7823878291994, 469.8231420969034, 349.8258450642994, 325.72720862796047, 319.24752100121395, 313.3089817308656, 288.98318763380786, 281.43504483214133, 251.53981709121803, 236.75874194328898, 223.75181732539758, 217.47617941422757, 215.61639439714403, 207.38289426976294, 203.23805677924895, 201.83872366765868, 174.29341925943493, 172.37090468436028, 171.40021028089814, 169.98442916074407, 160.42650913667325, 152.301046462752, 139.31893592644664, 139.28086213033006, 136.9075002654858, 131.4003738260287, 129.68021350042227, 128.43832806685248, 128.36896528332707, 1130.1616008866706, 932.3228035750277, 408.48034996294797, 400.18413224931214, 391.26383281807443, 371.62764091720817, 357.64915653528425, 354.0201318926288, 337.1350259147021, 299.2097488664697, 278.7050659034493, 270.69671930407475, 253.89076983523248, 248.6882725130166, 236.61699462253094, 231.82984275345862, 229.80875530729145, 213.55734838425113, 195.76316902475662, 191.5929168799292, 180.16188452947577, 176.6916253969892, 161.81051896488836, 161.62104064182404, 151.8809868202851, 145.48244380305704, 143.7065744321479, 143.23456776088472, 142.48217600608197, 138.39238936768294, 1156.8679436343434, 1015.0548201433077, 670.2725758698035, 567.7998611996393, 431.7240439495751, 397.0676075363528, 361.2966323424597, 357.67319579237426, 286.89612823272347, 251.47058797569218, 238.48721522937123, 236.66064029652222, 231.878018569486, 201.09142421466936, 165.11576389509656, 163.4176602117749, 157.74163979462307, 156.87356808235236, 152.61009938204742, 138.35317585823577, 137.50028306311322, 129.04589254339243, 101.46155949808198, 94.66444198703145, 90.16371257099436, 88.71863355397893, 85.91616727968854, 85.03413519280852, 84.38051702082315, 78.29390013243584, 1072.9793211317874, 990.270359112873, 635.8391691782921, 424.8579906847302, 416.0812646727549, 374.8832335062875, 362.67349522796104, 326.9341284879415, 284.36673831250795, 251.01576833866235, 247.4075920213605, 163.67352065533734, 155.7875932705965, 154.58793325141065, 151.16984632757791, 144.37490660293892, 137.52793481894273, 124.184575584521, 121.95191045494516, 121.17309041834588, 104.00311849112555, 99.19779712653772, 93.78704613923313, 87.58005157137929, 84.12073233753146, 80.11646003051409, 75.03662817550014, 74.05820283052508, 69.8104077930346, 69.21229316249149, 784.087556039397, 705.5346798113527, 503.7799363399388, 369.951870643846, 365.5403881687618, 342.42343904106497, 233.59801539098714, 222.39642061112485, 209.28793740221613, 199.8691298374336, 191.39470797789454, 190.2487121239091, 184.9542761878533, 130.73263335822017, 112.96365357616595, 100.32142973292474, 100.10522785831104, 91.10825936345812, 89.74682412798614, 86.2445936456092, 82.29464255150197, 81.70040463917181, 80.98814751358532, 80.20473856476836, 78.08173592020565, 73.26135638306357, 71.34127783673594, 69.05331345887117, 68.63579391937519, 67.52737846625315, 523.5925887205672, 500.79024435198306, 290.294755007793, 252.98749608397173, 251.7843348733908, 248.43538667711312, 207.39119464436635, 185.76666865844643, 175.00807610870132, 170.07504952532642, 127.67113376484816, 123.15431702869384, 122.84117274370541, 117.85717701101355, 112.29415703370364, 111.90121756493079, 105.9175179563148, 105.0474987597434, 104.66819991709369, 100.93353277503566, 96.78390435640071, 93.70528304763472, 89.129776452038, 81.32987573377581, 78.87969945313172, 75.9377522225771, 64.2644405627223, 63.79318570249539, 63.39339240561858, 61.20373521462139, 604.1763624295443, 538.0020081296361, 430.1212599019687, 337.66478222563785, 310.82865476719996, 233.78408103646564, 213.15284639221687, 190.46111487298467, 189.58541598576062, 153.2190779194409, 129.22984302916103, 121.0781165935778, 115.38947608834559, 114.4851100477196, 114.30108442454699, 109.85402022635519, 108.39272279648073, 85.26684458000443, 75.60813081128909, 75.58142455777187, 72.99845704349276, 64.8443344281904, 61.97236680808086, 60.49838343117289, 60.44894896471061, 60.327921502882084, 54.646628239981425, 54.584318594282415, 52.07243858636391, 50.87690220504477, 448.20200133174757, 336.87198045141025, 250.02608865649145, 229.3230924605319, 183.96623447932447, 172.13507230470867, 160.42130933591383, 151.8764927906885, 139.48320779317808, 138.01628962749103, 122.5840007457879, 119.72231442497628, 115.01308351489784, 113.37300801070488, 112.47872888620624, 106.80331796320749, 99.34833573613875, 96.69040860396558, 90.61294770491905, 84.79852012189568, 84.29952104329494, 80.0379776542561, 76.13473554970413, 75.00693482285136, 71.40264252711762, 70.69907309188052, 69.96025759681231, 69.52333911991985, 69.10420593465626, 68.43106610642823, 735.4541800375935, 441.11165632969517, 385.29700491649015, 248.26611647168238, 159.99388812892215, 123.05142887887651, 122.61045289578834, 120.46104691454578, 118.37985793905797, 116.15909867858224, 114.93929027906212, 114.01365519743626, 93.60619513119178, 90.85926046348834, 89.0156122787202, 86.96412459287666, 83.16842246526566, 79.88598739854073, 76.05154661990552, 73.30876109592563, 73.295738629171, 73.06540639777452, 68.25083087609349, 67.26426438032614, 57.438010264098416, 55.13632310605845, 51.7247318720851, 51.084017568720256, 48.02891590413967, 47.47613723020255], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -1.6991, -3.2311, -3.2876, -3.3666, -3.524, -3.9477, -4.0734, -4.0841, -4.0941, -4.3577, -4.5279, -4.6037, -4.6042, -4.6127, -4.7137, -4.9581, -4.9906, -5.033, -5.0348, -5.0352, -5.0548, -5.0708, -5.0726, -5.0787, -5.0945, -5.1162, -5.1375, -5.1531, -5.1663, -5.2179, -2.4463, -3.0704, -3.4695, -3.7651, -3.8366, -3.8568, -3.8756, -3.9567, -3.9832, -4.0959, -4.1567, -4.2134, -4.242, -4.2506, -4.2897, -4.31, -4.3169, -4.4644, -4.4755, -4.4812, -4.4896, -4.5478, -4.6, -4.6897, -4.69, -4.7073, -4.7486, -4.7619, -4.7716, -4.7721, -2.584, -2.7766, -3.6031, -3.6236, -3.6462, -3.6979, -3.7363, -3.7465, -3.7956, -3.9152, -3.9864, -4.0157, -4.08, -4.1008, -4.1507, -4.1712, -4.18, -4.2537, -4.3411, -4.3627, -4.4245, -4.4441, -4.5325, -4.5337, -4.5962, -4.6395, -4.6519, -4.6552, -4.6605, -4.6898, -2.4294, -2.5603, -2.9758, -3.1419, -3.4164, -3.5003, -3.5949, -3.605, -3.8262, -3.9584, -4.0116, -4.0194, -4.0398, -4.1829, -4.381, -4.3914, -4.427, -4.4325, -4.4603, -4.559, -4.5652, -4.6291, -4.8715, -4.9415, -4.9907, -5.0071, -5.0395, -5.05, -5.0578, -5.1335, -2.3459, -2.4262, -2.8697, -3.2736, -3.2945, -3.3991, -3.4323, -3.5363, -3.6762, -3.8014, -3.8159, -4.231, -4.2806, -4.2884, -4.3109, -4.3572, -4.4061, -4.509, -4.5272, -4.5336, -4.6877, -4.7354, -4.792, -4.8612, -4.902, -4.9513, -5.0177, -5.0309, -5.0907, -5.0994, -2.4587, -2.5644, -2.9017, -3.2111, -3.2232, -3.2887, -3.6724, -3.7217, -3.7827, -3.829, -3.8725, -3.8786, -3.9069, -4.256, -4.4032, -4.5229, -4.5251, -4.6202, -4.6354, -4.6756, -4.723, -4.7303, -4.7392, -4.749, -4.7762, -4.8407, -4.8676, -4.9006, -4.9068, -4.9233, -2.7059, -2.7505, -3.2972, -3.4352, -3.44, -3.4534, -3.6347, -3.7454, -3.8053, -3.8341, -4.1227, -4.159, -4.1615, -4.2033, -4.252, -4.2555, -4.311, -4.3193, -4.3229, -4.3596, -4.402, -4.4346, -4.4852, -4.5778, -4.6087, -4.6472, -4.8163, -4.8238, -4.8302, -4.8658, -2.4943, -2.6105, -2.8347, -3.0773, -3.1604, -3.4462, -3.539, -3.6521, -3.6567, -3.8709, -4.0423, -4.1079, -4.1564, -4.1644, -4.166, -4.206, -4.2195, -4.4618, -4.5835, -4.5838, -4.619, -4.7391, -4.7851, -4.8095, -4.8103, -4.8124, -4.9129, -4.9141, -4.962, -4.9857, -2.7256, -3.0118, -3.3109, -3.3977, -3.6191, -3.6859, -3.7567, -3.8118, -3.8975, -3.9081, -4.0275, -4.0513, -4.0919, -4.1063, -4.1143, -4.1665, -4.2395, -4.2669, -4.3324, -4.3995, -4.4054, -4.4579, -4.5085, -4.5236, -4.5735, -4.5836, -4.5942, -4.6005, -4.6067, -4.6166, -2.1741, -2.6862, -2.8218, -3.2626, -3.704, -3.9683, -3.9719, -3.9898, -4.0073, -4.0264, -4.037, -4.0452, -4.2442, -4.2743, -4.295, -4.3186, -4.3637, -4.4044, -4.4542, -4.4914, -4.4916, -4.4948, -4.5638, -4.5786, -4.7389, -4.7805, -4.8455, -4.8582, -4.921, -4.9328], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.6363, 1.6355, 1.6354, 1.6353, 1.6351, 1.6344, 1.6341, 1.634, 1.634, 1.6333, 1.6327, 1.6324, 1.6324, 1.6323, 1.6319, 1.6306, 1.6304, 1.6302, 1.6302, 1.6302, 1.63, 1.6299, 1.6299, 1.6299, 1.6298, 1.6296, 1.6295, 1.6294, 1.6293, 1.6289, 2.0245, 2.0239, 2.0233, 2.0226, 2.0224, 2.0224, 2.0223, 2.0221, 2.022, 2.0216, 2.0214, 2.0211, 2.021, 2.021, 2.0208, 2.0207, 2.0207, 2.02, 2.0199, 2.0199, 2.0198, 2.0195, 2.0192, 2.0186, 2.0186, 2.0185, 2.0182, 2.0181, 2.0181, 2.0181, 2.031, 2.0308, 2.0296, 2.0295, 2.0295, 2.0293, 2.0293, 2.0292, 2.029, 2.0288, 2.0285, 2.0284, 2.0282, 2.0281, 2.0279, 2.0279, 2.0278, 2.0275, 2.0271, 2.027, 2.0267, 2.0266, 2.0261, 2.0261, 2.0258, 2.0255, 2.0254, 2.0254, 2.0254, 2.0252, 2.1622, 2.1621, 2.1616, 2.1614, 2.1609, 2.1607, 2.1605, 2.1604, 2.1598, 2.1593, 2.1591, 2.1591, 2.159, 2.1584, 2.1574, 2.1574, 2.1572, 2.1571, 2.157, 2.1563, 2.1563, 2.1558, 2.1539, 2.1532, 2.1527, 2.1526, 2.1522, 2.1521, 2.152, 2.1512, 2.321, 2.3209, 2.3204, 2.3197, 2.3197, 2.3194, 2.3193, 2.3191, 2.3186, 2.3182, 2.3182, 2.3162, 2.316, 2.3159, 2.3158, 2.3155, 2.3152, 2.3143, 2.3143, 2.3143, 2.313, 2.3126, 2.312, 2.3113, 2.3109, 2.3103, 2.3094, 2.3094, 2.3086, 2.3085, 2.5219, 2.5218, 2.5213, 2.5206, 2.5206, 2.5204, 2.5191, 2.5189, 2.5187, 2.5184, 2.5182, 2.5182, 2.5181, 2.516, 2.5149, 2.5138, 2.5138, 2.5129, 2.5127, 2.5123, 2.5118, 2.5117, 2.5116, 2.5115, 2.5111, 2.5104, 2.51, 2.5096, 2.5095, 2.5093, 2.6784, 2.6784, 2.677, 2.6766, 2.6765, 2.6765, 2.6758, 2.6752, 2.6749, 2.6748, 2.673, 2.6727, 2.6727, 2.6724, 2.672, 2.6719, 2.6715, 2.6714, 2.6714, 2.671, 2.6706, 2.6703, 2.6698, 2.6688, 2.6685, 2.668, 2.6658, 2.6657, 2.6656, 2.665, 2.7469, 2.7467, 2.7463, 2.7457, 2.7455, 2.7445, 2.7441, 2.7436, 2.7435, 2.7424, 2.7412, 2.7407, 2.7404, 2.7403, 2.7403, 2.74, 2.7398, 2.7375, 2.7361, 2.7361, 2.7356, 2.734, 2.7333, 2.733, 2.7329, 2.7329, 2.7313, 2.7313, 2.7304, 2.73, 2.8143, 2.8136, 2.8126, 2.8123, 2.8113, 2.811, 2.8106, 2.8102, 2.8097, 2.8096, 2.8088, 2.8086, 2.8082, 2.8081, 2.8081, 2.8076, 2.807, 2.8067, 2.8061, 2.8053, 2.8053, 2.8047, 2.8041, 2.8039, 2.8033, 2.8031, 2.803, 2.8029, 2.8028, 2.8027, 2.8705, 2.8696, 2.8693, 2.868, 2.8659, 2.8642, 2.8642, 2.864, 2.8639, 2.8637, 2.8636, 2.8636, 2.8618, 2.8615, 2.8613, 2.861, 2.8605, 2.8601, 2.8595, 2.859, 2.859, 2.859, 2.8581, 2.8579, 2.8555, 2.8548, 2.8537, 2.8534, 2.8522, 2.852]}, "token.table": {"Topic": [9, 1, 3, 1, 4, 5, 7, 2, 2, 2, 1, 7, 8, 5, 7, 8, 4, 2, 3, 7, 10, 9, 4, 10, 3, 1, 10, 6, 4, 6, 6, 6, 10, 7, 7, 6, 3, 3, 1, 7, 2, 8, 9, 10, 1, 8, 1, 4, 8, 9, 10, 7, 5, 10, 10, 4, 5, 3, 3, 7, 6, 5, 7, 2, 2, 7, 8, 5, 10, 4, 2, 3, 9, 6, 8, 10, 5, 1, 7, 7, 1, 6, 5, 6, 5, 3, 6, 5, 1, 10, 4, 7, 3, 3, 2, 6, 10, 1, 4, 6, 3, 1, 10, 7, 5, 5, 8, 3, 9, 3, 6, 8, 3, 10, 9, 7, 6, 3, 10, 2, 9, 5, 3, 5, 4, 2, 3, 10, 5, 1, 9, 4, 2, 9, 4, 2, 1, 10, 5, 8, 1, 5, 5, 10, 10, 9, 4, 1, 4, 9, 9, 7, 2, 5, 8, 9, 5, 7, 6, 4, 6, 8, 6, 1, 6, 2, 8, 8, 2, 9, 8, 6, 7, 7, 1, 7, 4, 10, 8, 7, 5, 2, 2, 6, 3, 9, 10, 3, 8, 1, 7, 7, 10, 2, 2, 4, 3, 4, 1, 1, 8, 5, 8, 2, 6, 3, 2, 3, 10, 6, 3, 10, 2, 3, 2, 5, 8, 1, 7, 8, 6, 4, 4, 3, 1, 3, 4, 1, 4, 10, 9, 3, 7, 8, 9, 8, 1, 6, 4, 10, 9, 5, 9, 9, 8, 5, 7, 2, 8, 4, 9, 6, 8, 9, 9, 5, 3, 9, 10, 9, 4, 3, 4, 9, 2, 5, 1, 6, 9, 2, 10, 1, 1, 7, 1, 9, 7, 10, 5, 4, 5, 4, 5, 6, 2, 6, 4, 7, 2, 6, 8, 10, 8, 8, 2, 6, 1, 9, 8, 4], "Freq": [0.9862742415508764, 0.9892685188963304, 0.9899388299165526, 0.9970473296347099, 0.9929810816703226, 0.9916081572921627, 0.9942398309202874, 0.9965872487329146, 0.9939083417797402, 0.9958445849616068, 0.9948730197367552, 0.9907510145600833, 0.9827642374625981, 0.9943106352967577, 0.9988682255376929, 0.9869790562948036, 0.9968764715012238, 0.9977674305102615, 0.9961015699298365, 0.9780199110231703, 0.9914553704214889, 0.9924785298946965, 0.9962134462986069, 0.9961726472884397, 0.9970410314805425, 0.9936033356422055, 0.9816730307889684, 0.9921901534128603, 0.9954508929257025, 0.9914693483641956, 0.9827827869242786, 0.9811991335534325, 0.9923742089587634, 0.9982474852599039, 0.9876510405545254, 0.9849792096286744, 0.9969053298546482, 0.9960856922594081, 0.9965690670027227, 0.9919004677315555, 0.9947546855294154, 0.9851425887020391, 0.9942289107775457, 0.9883438114973214, 0.9955905022126746, 0.9904832893058338, 0.9932992530399509, 0.9992497470094843, 0.9910956940534786, 0.9911401462698595, 0.9859511286780311, 0.9906270680838181, 0.9986802178617329, 0.9854184565541954, 0.9889093515972875, 0.9878385875217793, 0.9974013137226803, 0.9983111027057646, 0.9967698705781936, 0.9936160178772258, 0.9861461082101055, 0.9981429709178563, 0.9936789697940595, 0.9918307551746675, 0.9933714349927798, 0.9803323243197448, 0.9917620372164178, 0.9961967713841255, 0.9899710200117197, 0.9932425356078936, 0.9978539497346368, 0.9964915233593936, 0.9905833247944914, 0.9958430443749668, 0.979404871070418, 0.9974798754153431, 0.9971427623898985, 0.9901940932210048, 0.9927269850445047, 0.983648373715349, 0.9961615682056296, 0.9956515053718374, 0.9866770972340735, 0.9974270419495666, 0.9958849748411313, 0.9964204662195233, 0.9889593392677216, 0.9904768312216458, 0.9947000240924739, 0.9918279443279874, 0.9979570593379496, 0.998422005298834, 0.9981849348071363, 0.9935508860128199, 0.9996639745278824, 0.9984518312785436, 0.9861732381964433, 0.9984383085011105, 0.9836393865602593, 0.9914271582586998, 0.9941994923872365, 0.9992788750692273, 0.9821472757640373, 0.9924733907769512, 0.9933768984948983, 0.9979805236019041, 0.9966461316228655, 0.99661588544197, 0.984021147197605, 0.997470207794165, 0.9974399808577731, 0.9945914567329045, 0.9961574270320388, 0.9950212002209372, 0.9939667519088189, 0.9931523549888243, 0.9943959412474108, 0.9913807973858259, 0.9935239849206329, 0.9965977687426795, 0.9932355395068361, 0.9824844242677329, 0.998972181601499, 0.986064536175352, 0.9983229010296687, 0.9960923079455668, 0.9974261996751719, 0.9900214559877955, 0.9857112002441287, 0.9895152240834473, 0.9974115376106918, 0.9893364973241174, 0.9958220740317301, 0.992636451608473, 0.9972084910456827, 0.9933316859395159, 0.993706508905583, 0.9905429511630943, 0.9883913041242045, 0.9980529486045866, 0.991976299110933, 0.9951937476210528, 0.9987171593078772, 0.9823217740430096, 0.9937879618993866, 0.9803558737117478, 0.9989608244575808, 0.9993468388861735, 0.9952983892167682, 0.9958960736377347, 0.9901119907049982, 0.9913374295959879, 0.9942087097882762, 0.9990872879723773, 0.9980312361234237, 0.9878894629789197, 0.9903549191055157, 0.9960966605098777, 0.9878357969826101, 0.9963615852130022, 0.9927129229818851, 0.9892951197462851, 0.9916785453385948, 0.9986062889288454, 0.993721029289556, 0.9976392680073491, 0.9981375383093465, 0.9969121254252775, 0.9984541909753961, 0.9928595957558373, 0.9886170421645837, 0.9842682037206524, 0.9947432614949258, 0.987566294209001, 0.9972892051749284, 0.993291930032266, 0.994572497465112, 0.985988678029741, 0.9919568066983837, 0.9884752949940644, 0.9976439770377922, 0.9893356990117854, 0.9966399498588017, 0.9847463734017842, 0.9950832142861945, 0.9845844789245299, 0.994900164026907, 0.9966333195086372, 0.9920435631385155, 0.9911832043706361, 0.9888475810730691, 0.9900283322105635, 0.9889135365026646, 0.990532974446898, 0.9914574029990648, 0.9944313876899021, 0.9971184353636189, 0.9964111695864614, 0.9965076168951383, 0.9942282694772355, 0.9922258627895868, 0.9861850378847565, 0.9779882769072585, 0.997141245224569, 0.9948404751297338, 0.9966838348982497, 0.9973414048652652, 0.9949909377333853, 0.9966337529232254, 0.9992421636715354, 0.9964807463222626, 0.9785771574317174, 0.9893426952659757, 0.9973901699544973, 0.9978104295582617, 0.9961612539326082, 0.9843096712589318, 0.9997322859485976, 0.9873243656944888, 0.9925730889883181, 0.9986129660762754, 0.9981178466815696, 0.9902194087714885, 0.9972323885398312, 0.9983471467067945, 0.9988240561314846, 0.9913249265107716, 0.991863162128126, 0.9981286560011635, 0.9787797119272776, 0.9952359138041269, 0.9959568534412642, 0.9803244134446607, 0.9863222171545643, 0.9947477618267332, 0.987958381167432, 0.9935440459243869, 0.9934364227228208, 0.9834738066407833, 0.9911093526851592, 0.985095691979342, 0.9949444416332405, 0.9924724685760972, 0.9942304438408895, 0.9871511411416823, 0.9921943784939978, 0.996884894074982, 0.9908037463960302, 0.9923073088239102, 0.9960022345538214, 0.9937007249637495, 0.9907367004434735, 0.988167097206039, 0.9934059207719191, 0.9957438273801035, 0.9922613778078274, 0.9985811742778838, 0.9864282000684962, 0.9812045163658116, 0.9973181705387838, 0.9918998577278392, 0.997392433187163, 0.9981014054347187, 0.9965357278426334, 0.9988819555295948, 0.9904611697632709, 0.9894712232276491, 0.9877988626246876, 0.9870314357674066, 0.9920467744433399, 0.9993824495802441, 0.9972449361251939, 0.9907516268679349, 0.9955398608295274, 0.993828128629707, 0.9911915802625479, 0.991946311357981, 0.9793906622341745, 0.9959533684063548, 0.9918951892014638, 0.9903188866909656, 0.9973112701311072, 0.9879251640536957, 0.9868280412625433, 0.9925790700249578, 0.9985216731549994, 0.9870933379093273, 0.9958729482313319, 0.9982479745607473, 0.9938460982596387, 0.9957626799894117, 0.9885906275009357, 0.9973340464127395, 0.9973931539626192, 0.9967952949189487, 0.9855690241787978, 0.9991418105219567, 0.98657544365425, 0.9975789552986069, 0.9985912972962879], "Term": ["accept", "action", "add", "air", "allow", "almost", "alone", "also", "always", "answer", "appear", "appearance", "approach", "arm", "ask", "attention", "away", "back", "bad", "bar", "bear", "beg", "begin", "believe", "bella", "better", "blood", "blow", "boat", "boot", "break", "brewer", "bright", "bring", "business", "buy", "call", "can", "care", "carriage", "case", "catch", "certain", "change", "child", "close", "clothe", "come", "comfortable", "company", "confess", "corner", "could", "course", "creature", "cross", "cry", "day", "dear", "dinner", "direction", "do", "door", "draw", "dress", "drop", "ease", "else", "enjoyment", "enough", "eugene", "even", "event", "ever", "explain", "eye", "face", "fall", "family", "far", "feel", "feeling", "fellow", "find", "fire", "first", "flush", "follow", "foot", "force", "friend", "get", "give", "glance", "go", "good", "grateful", "great", "guest", "hair", "half", "hand", "happy", "hard", "hat", "head", "hear", "heart", "hide", "hold", "home", "honour", "hope", "hour", "house", "however", "husband", "indeed", "induce", "keep", "kind", "kiss", "know", "knowledge", "lady", "last", "laugh", "lay", "lean", "learn", "leave", "less", "let", "lie", "life", "light", "lightwood", "likewise", "lip", "little", "live", "long", "look", "lose", "love", "low", "make", "man", "manner", "many", "marriage", "marry", "master", "may", "mean", "meet", "member", "mind", "miss", "moment", "money", "mother", "move", "much", "must", "name", "never", "next", "night", "noble", "notice", "observe", "occasion", "offer", "old", "open", "opinion", "order", "other", "paper", "part", "pass", "pay", "perhaps", "person", "piece", "place", "podsnap", "point", "poor", "possible", "pound", "power", "present", "pretty", "pursue", "put", "question", "quite", "rather", "really", "relation", "remark", "repeat", "reply", "rest", "resume", "retort", "return", "riderhood", "right", "ring", "rise", "river", "room", "round", "save", "say", "school", "scream", "see", "seem", "sense", "set", "shake", "shall", "short", "show", "side", "silence", "sir", "sit", "sleep", "slight", "sloppy", "small", "smile", "society", "sometimes", "soon", "sort", "speak", "spirit", "stand", "stare", "state", "still", "stop", "story", "strike", "strong", "subject", "sum", "suppose", "sure", "table", "take", "talk", "tear", "tell", "thank", "thing", "think", "throw", "time", "tippin", "tone", "touch", "true", "try", "turn", "twemlow", "understand", "use", "veneer", "veneering", "venus", "visit", "voice", "wait", "walk", "want", "watch", "water", "wave", "way", "wear", "wegg", "well", "wife", "window", "wish", "woman", "word", "work", "world", "would", "write", "yet", "young"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [3, 8, 6, 5, 7, 4, 10, 1, 9, 2]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el591406496063327521145522333", ldavis_el591406496063327521145522333_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.2.2/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el591406496063327521145522333", ldavis_el591406496063327521145522333_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.2.2/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el591406496063327521145522333", ldavis_el591406496063327521145522333_data);
            })
         });
}
</script>




```python
# Export the visualization as a html file.
pyLDAvis.save_html(visualization, 'drive/My Drive/Colab Notebooks/Tutorials/TopicModeling/LDAModel.html')
```

References: \
[Topic Modeling with Gensim (Python)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) by Selva Prabhakaran. \
[Generating and Visualizing Topic Models with Tethne and MALLET](https://diging.github.io/tethne/api/tutorial.mallet.html) by ASU Digital Innovation Group. \
[Colab + Gensim + Mallet](https://github.com/polsci/colab-gensim-mallet) by 
Geoff Ford.
