import json
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load data from JSON file
with open('data/nyt_contents_1.json', 'r') as f:
    data = json.load(f)

articles = []
max_len = 0
avg_len = 0
count = 0

# convert data to label, content format
for article in data:

    new = {}

    if article != None and article['content'] != "":

        new['label'] = article['view'] + ": " + article['title']
        new['content'] = article['content']

        avg_len += len(new['content'].split())
        count += 1
        if len(new['content'].split()) > max_len:
            max_len = len(new['content'].split())
        articles.append(new)

print(avg_len / count)

tokenized_articles = []

for article in articles:
    # dictionary to hold tokenized article content
    tokenized = {}

    # Create a TextVectorization layer
    vectorizer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=6)

    label = [article['label']]
    content = [article['content']]

    # Adapt the TextVectorization layer to the text
    tokenized['label'] = vectorizer.adapt(label)
    tokenized['content'] = vectorizer.adapt(content)

    # # save tokenized inputs to dictionary
    # tokenized['label'] = tokenizer.encode_plus(article['label'], return_tensors='pt')
    # tokenized['content'] = tokenizer.encode_plus(article['content'], return_tensors='pt')

    # save tokenized article to article list
    tokenized_articles.append(tokenized)

    print(tokenized_articles[0])