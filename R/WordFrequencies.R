#Script Name: WordFrequencies.R
#Location: ~~~
#Created by Hoyeol Kim
#Creation Date: 12/20/2020
#Purpose: Word Frequencies with R
#Last Modified: 12/20/2020

ls()
rm(list=ls())

# Load a text file
install.packages("tidytext")
install.packages("readtext")
install.packages("magrittr")

library(tidytext)
library(readtext)
library(magrittr)

omf <- readtext("/Volumes/SanDisk1TB/Desktop/R/Text/OMF.txt")

summary(omf)

head(omf, 20) 

install.packages("tidyr")

library("tidyr")

token_omf <- omf %>% unnest_tokens(word, text)

library("ggplot2")
library("dplyr")

token_omf %>%
  count(word, sort = TRUE) %>%
  filter(n > 500) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col(fill = "lightblue") +
  xlab(NULL) +
  coord_flip() +
  theme_bw() +
  labs(y = "Word Frequencies for Our Mutual Friend")

data("stop_words")
stop_words

token_omf %>%
  count(word, sort = TRUE) %>%
  filter(n > 200) %>%
  anti_join(stop_words) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col(fill = "purple") +
  xlab(NULL) +
  coord_flip() +
  theme_bw() +
  labs(y = "Word Frequencies for Our Mutual Friend")

# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes

# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

getwd()
a <- readLines("/Volumes/SanDisk1TB/Desktop/R/Text/OMF.txt")

# Load the data as a corpus
docs <- Corpus(VectorSource(a))

inspect(docs)

toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")

# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
docs <- tm_map(docs, removeWords, c("blabla1", "blabla2")) 
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
# Text stemming
# docs <- tm_map(docs, stemDocument)

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 5,
          max.words=200, random.order=TRUE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))
