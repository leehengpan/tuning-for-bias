import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
import json
from embedding import positional_encoding

np.random.seed(2470)
CONTENT_SEQ_LEN = 256
TITLE_SEQ_LEN = 16
START_TOKEN = '<start>'
END_TOKEN = '<end>'
PAD_TOKEN = ''
CONTENT_VECTORIZER = TextVectorization(max_tokens=100000, split='whitespace', output_mode='int',
                                       standardize='lower', output_sequence_length=CONTENT_SEQ_LEN)
TITLE_VECTORIZER = TextVectorization(max_tokens=15000, split='whitespace', output_mode='int',
                                     standardize='lower', output_sequence_length=TITLE_SEQ_LEN)
GLOVE_EMBED_SZ = 100
GLOVE_EMBEDDINGS = '../data/glove.6B/glove.6B.100d.txt'


def pad_and_trim(text, window_size=16, start_token=START_TOKEN, end_token=END_TOKEN, pad_token=PAD_TOKEN):
    words = text.split()
    num_words = len(words)
    trim_length = max((num_words - (window_size - 2)), 0)
    pad_length = max(0, ((window_size - 2) - num_words))
    padded_text = ' '.join([start_token] + words[:num_words - trim_length] + [end_token] + [pad_token] * pad_length)
    return padded_text


def train_test_split(input_file, test_split=0.05,
                     content_seq_len=CONTENT_SEQ_LEN, title_seq_len=TITLE_SEQ_LEN,
                     shuffle=True):
    """Read data from input_file and split into train and test arrays"""

    with open(input_file) as f:
        data = json.load(f)

    # separate content and title data into separate lists
    content_arr = [pad_and_trim(item['content'], content_seq_len) for item in data]
    title_arr = [pad_and_trim(item['title'], title_seq_len) for item in data]

    num_samples = len(content_arr)
    num_test_samples = int(test_split * num_samples)

    # find random indices to create train and test arrays
    idx = np.arange(0, num_samples)

    if shuffle:
        np.random.shuffle(idx)

    # create train and test sets for content and titles
    temp_content_arr = np.array(content_arr)[idx].tolist()
    temp_title_arr = np.array(title_arr)[idx].tolist()

    train_content = temp_content_arr[:num_samples-num_test_samples]
    test_content = temp_content_arr[num_samples-num_test_samples:]

    train_title = temp_title_arr[:num_samples-num_test_samples]
    test_title = temp_title_arr[num_samples-num_test_samples:]

    return train_content, train_title, test_content, test_title


def vectorize_data(train_content, train_title,
                   content_vectorizer=CONTENT_VECTORIZER, title_vectorizer=TITLE_VECTORIZER):
    """Vectorize the train content and train title data"""

    train_content_ds = tf.data.Dataset.from_tensor_slices(train_content).batch(128)
    train_title_ds = tf.data.Dataset.from_tensor_slices(train_title).batch(128)

    content_vectorizer.adapt(train_content_ds)
    title_vectorizer.adapt(train_title_ds)

    # create dictionaries that map unique words to indexes for both content and title data
    content_vocab = content_vectorizer.get_vocabulary()
    content_word_index = dict(zip(content_vocab, range(len(content_vocab))))
    content_index_word = dict(zip(range(len(content_vocab)), content_vocab))

    title_vocab = title_vectorizer.get_vocabulary()
    title_word_index = dict(zip(title_vocab, range(len(title_vocab))))
    title_index_word = dict(zip(range(len(title_vocab)), title_vocab))

    return content_vocab, content_word_index, content_index_word, title_vocab, title_word_index, title_index_word


def build_glove_embed_index(path_to_glove=GLOVE_EMBEDDINGS,
                            start_token=START_TOKEN, end_token=END_TOKEN, pad_token=PAD_TOKEN,
                            embedding_size=GLOVE_EMBED_SZ):
    embeddings_index = {}
    with open(path_to_glove) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    embeddings_index[start_token] = np.random.normal(size=(embedding_size), scale=0.1)
    embeddings_index[end_token] = np.random.normal(size=(embedding_size), scale=0.1)
    embeddings_index[pad_token] = np.zeros(shape=(embedding_size))

    print('Unique words in glove: {}'.format(len(embeddings_index)))
    return embeddings_index

def build_embedding_init(word_index, embeddings_index, embedding_size=GLOVE_EMBED_SZ):
    hits = 0
    misses = 0
    num_tokens = len(word_index)
    embedding_init = np.zeros((num_tokens, embedding_size))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_init[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f'Hits: {hits}; Misses: {misses}')
    return embedding_init, num_tokens

# last four functions not used
def build_start_stop_embeddings(emb_sz=GLOVE_EMBED_SZ, seed=2470):
    np.random.seed(seed)
    start_embedding = np.random.normal(size=(emb_sz))
    stop_embedding = np.random.normal(size=(emb_sz))

    return start_embedding, stop_embedding

def create_embeddings(data, glove_index, window_size, embedding_size=100, dataset_name=None,
                      add_positional_embedding=True):
    embedding_matrix = np.zeros(shape=(len(data), window_size, embedding_size))

    start_embedding_vec, end_embedding_vec = build_start_stop_embeddings(embedding_size)

    for j, article in enumerate(data):
        words = article.split()
        for i in range(-1, window_size - 1):
            try:
                if i == -1:
                    embedding_matrix[j][i + 1] = start_embedding_vec
                elif i == window_size - 2:
                    embedding_matrix[j][i + 1] = end_embedding_vec
                else:
                    embedding_matrix[j][i + 1] = glove_index.get(words[i], np.zeros(embedding_size))
            except IndexError:
                embedding_matrix[j][window_size - 1] = end_embedding_vec

    if add_positional_embedding:
        embedding_matrix += positional_encoding(window_size, embedding_size)

    if dataset_name:
        print(f"Shape of {dataset_name} embedding: {embedding_matrix.shape}")
    return embedding_matrix


def create_token_labels(data, vectorizer, dataset_name=None):
    labels = []
    for title in data:
        labels.append(vectorizer(title).numpy())
    labels = np.array(labels).reshape(len(labels), -1, 1)

    if dataset_name:
        print(f"Shape of {dataset_name} token labels: {labels.shape}")
    return labels


def save_to_pickle(obj, filepath):
    with open(filepath, 'wb+') as f:
        pickle.dump(obj, f)

