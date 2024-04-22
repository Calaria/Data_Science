from typing import Set
import re

def tokenize(text: str)->Set:
    text=text.lower()
    all_words=re.findall("[a-z0-9']+", text)
    return set(all_words)
assert tokenize("Data Science is science")=={"data","science","is"}

from typing import NamedTuple
class Message(NamedTuple):
    text:str
    is_spam:bool

from typing import List,Tuple,Dict,Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self,k:float = 0.5)-> None:
        self.k=k#smoothing factor
        self.tokens:Set[str]=set()
        self.token_spam_counts:Dict[str, int] = defaultdict(int)
        self.token_ham_counts:Dict[str,int]=defaultdict(int)
        self.spam_messages = self.ham_messages =0
    
    def train(self, messages: Iterable[Message])->None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token]+=1
                else:
                    self.token_ham_counts[token]+=1
    def _probabilities(self,token:str)->Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        
        p_token_spam = (spam+self.k)/(self.spam_messages + 2*self.k)
        p_token_ham = (ham+self.k)/(self.ham_messages + 2*self.k)
        
        return p_token_spam, p_token_ham
    
    def predict(self, text: str)->float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham=0.0
        for token in self.tokens:
            prob_if_spam,prob_if_ham=self._probabilities(token)
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            
            else:
                log_prob_if_spam+=math.log(1.0-prob_if_spam)
                log_prob_if_ham+=math.log(1.0-prob_if_ham)
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham=math.exp(log_prob_if_ham)
        return prob_if_spam/(prob_if_spam+prob_if_ham)
"""           
messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)
text="spam hello ham ham ham ham"
print(model.predict(text))
"""

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
random.seed(0)
train_messages, test_messages = split_data(data, 0.75)
model=NaiveBayesClassifier()
model.train(train_messages)
from collections import Counter
predictions=[(message, model.predict(message.text)) for message in test_messages]
confusion_matrix=Counter((message.is_spam, spam_probability>0.5)
                         for message, spam_probability in predictions)
# 步骤 1: 读取文本文件
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# 步骤 1: 逐行读取文本文件
def read_lines_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()  # 移除行尾的换行符

# 步骤 3: 使用模型对每条消息进行预测
def predict_spam_for_each_line(file_path, model):
    results = []
    for line in read_lines_from_file(file_path):
        if not line:
            continue  # 跳过空行
        match = re.search(pattern, line)
        if match:
            name = match.group(1)
        spam_probability = model.predict(line)  # 对每行（每条消息）进行预测
        results.append((name, spam_probability))
    return results

# 步骤 4: 解析和展示预测结果
pattern = r'@(\w+)'
file_path = 'email.txt'
results = predict_spam_for_each_line(file_path, model)  

for name, spam_probability in results:
    print(f"来自{name}的为垃圾邮件概率为: {100*spam_probability:.2f}%",end="\t")
    if (spam_probability > 0.60):
        print("该消息是垃圾邮件。")
    elif (spam_probability < 0.40):
        print("不是垃圾邮件。")
    else:
        print("可能是垃圾邮件。")
    
       

text="Discover the best ways to use pack offers with Experiences. Experiences are curated bundles of pack partner products, "
print(model.predict(text))
print(confusion_matrix)