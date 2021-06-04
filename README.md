# Twitter-Opinion-Mining

This project consists of two modules:
1. Get opinion of twitter users on a certain topic
2. Analyze a particular users tweets and get their general tweet sentiment i.e if the users tweets are mostly positive or negative.

A SVM classifier was trained on the [sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset containing 1.6M tweets. 

## Installation
Use Python3 to create a virtual environment and install all required packages by running:

    pip install -r requirements.txt

## Usage
1. Apply for a Twitter developer account [here](https://developer.twitter.com/en/apply-for-access).
2. Get consumer and access tokens through their dashboard.
3. Replace empty strings with your keys from line 23-26 in `sentiment.py`.
4. Execute `python3 sentiment.py`
5. Use the menu to peform desired function. 
