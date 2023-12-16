# IDEA EDU Course

Project implemented in the track 'Machine Learning Algorithms from Scratch' of hyperskill.org JetBrains Academy. 
Purpose of doing this project is to make progress in this track and improve ML knowledge.

## Technology / External Libraries

- Python 3.11.6,
- with packages: nltk, typing, unittest, unittest.mock, parameterized

## Repository Contents

The sources of main project tasks (6 stages) and pytest testing.

## Installation

Install the wheel file using pip after clone/fork:

    > pip install dist/spam-filter-3.11.6-py3-none-any.whl

## Program description

Spam filter is a Python-based project. We build a spam filter from scratch with Naive Bayes, one of the most common
algorithms. We learn about data preprocessing and feature extraction from texts with SpaCy and Pandas. We make use of
functions to perform repetitive tasks and functional decomposition. We implement our own algorithms and measure how
well they perform against the Multinomial Naive Bayes classification algorithm from the Scikit-learn library.

Have fun!

## Project progress

[//]: # (Project was completed on 29.10.23)

12.11.23 Project started and Initial Repo setup.

13.11.23 Stage 1 completed. Data preprocessing including lemmatization with en_core_sm model of SpaCy, stop words 
and special characters removal, lower case conversion and number replacement.

16.12.23 Stage 2 completed. Feature extraction from texts with Pandas. We use CountVectorizer to create a matrix of
word counts and print its first 200 rows and 50 columns together with the data. We use @patch on custom method with
MagicMock.return_value to mock the return value of the method.