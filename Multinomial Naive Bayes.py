import re
from collections import Counter
from typing import Tuple, Dict, Iterable
from typing import NamedTuple
import math
from collections import defaultdict




def tokenize(text: str) -> Counter:
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)
    return Counter(all_words)


class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor
        self.tokens: Counter[str] = Counter()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            for token, count in tokenize(message.text).items():
                self.tokens[token] += count
                if message.is_spam:
                    self.token_spam_counts[token] += count
                else:
                    self.token_ham_counts[token] += count

    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> str:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam) * text_tokens[token]
                log_prob_if_ham += math.log(prob_if_ham) * text_tokens[token]
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        p = prob_if_spam / (prob_if_spam + prob_if_ham)
        return p


from io import BytesIO
import requests
import tarfile
import os
BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]
OUTPUT_DIR = 'spam_data'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
for filename in FILES:
    local_file_path = f"{OUTPUT_DIR}/{filename}"
    if not os.path.exists(local_file_path):
        print(f"Downloading {filename}")
        content = requests.get(f"{BASE_URL}/{filename}").content
        fin = BytesIO(content)
        print(f"Extracting {filename} to {OUTPUT_DIR}")
        with open(local_file_path, 'wb') as file:
                file.write(content)
        with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
            tf.extractall(OUTPUT_DIR)
    else:
        print(f"Already downloaded {filename}")
import glob,re
from typing import List
path='spam_data/*/*'
data:List[Message]=[]
for filename in glob.glob(path):
    is_spam="ham" not in filename
    with open(filename, errors='ignore') as file:
        for line in file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break
from split_data import split_data
import random
random.seed(42)
train_messages, test_messages = split_data(data, 0.75)
print(len(train_messages), len(test_messages))
model=NaiveBayesClassifier()
model.train(train_messages)
from collections import Counter
predictions=[(message, model.predict(message.text)) for message in test_messages]
confusion_matrix=Counter((message.is_spam, spam_probability>0.5)
                         for message, spam_probability in predictions)
print(confusion_matrix)

#See the most common spam and ham words
"""
def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilities(token)
    return prob_if_spam / (prob_if_spam + prob_if_ham)
words=sorted(model.tokens,key=lambda t: p_spam_given_token(t,model))
print("spammiest_words",words[-10:])
print("hammiest_words",words[:10])
"""