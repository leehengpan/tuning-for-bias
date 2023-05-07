import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle
import json

# CHANGE WITH CARE
np.random.seed(2470)
CONTENT_VECTORIZER = TextVectorization(max_tokens=100000, split='whitespace', output_mode='int',
                                       standardize='lower_and_strip_punctuation', output_sequence_length=256)
TITLE_VECTORIZER = TextVectorization(max_tokens=15000, split='whitespace', output_mode='int',
                                     standardize='lower_and_strip_punctuation', output_sequence_length=16)
GLOVE_EMBED_SZ = 100


def train_test_split(input_file='../data/v2/nytfox_collate.json', test_split=0.05, shuffle=True):
    """Read data from input_file and split into train and test arrays"""

    with open(input_file) as f:
        data = json.load(f)

    # separate content and title data into separate lists
    content_arr = [item['content'] for item in data]
    title_arr = [item['title'] for item in data]

    num_samples = len(content_arr)
    num_test_samples = int(test_split * num_samples)

    # find random indices to create train and test arrays
    idx = np.arange(0, num_samples)
    if shuffle:
        np.random.shuffle(idx)

    # create train and test sets for content and titles
    temp_content_arr = np.array(content_arr)[idx].tolist()
    temp_title_arr = np.array(title_arr)[idx].tolist()

    train_content = temp_content_arr[:-num_test_samples]
    test_content = temp_content_arr[-num_test_samples:]

    train_title = temp_title_arr[:-num_test_samples]
    test_title = temp_title_arr[-num_test_samples:]

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


def build_glove_embed_index(path_to_glove='../data/glove.6B/glove.6B.100d.txt'):
    embeddings_index = {}
    with open(path_to_glove) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print('Unique words in glove: {}'.format(len(embeddings_index)))
    return embeddings_index


def build_start_stop_embeddings(emb_sz=GLOVE_EMBED_SZ, seed=2470):
    np.random.seed(seed)
    start_embedding = np.random.normal(size=(emb_sz))
    stop_embedding = np.random.normal(size=(emb_sz))

    return start_embedding, stop_embedding


def positional_encoding(window_size, embedding_size):
    depth = embedding_size / 2

    # setup position and depth arrays of size (256 [or 32 with title] x 1) and (1 x 50) respectively
    positions = np.arange(window_size).reshape(-1, 1)
    depths = np.arange(depth).reshape(1, -1) / depth
    angle_rates = 1 / (10000 ** depths)

    # multiply the two matrices
    angle_rads = positions * angle_rates

    # get 100-D positional encoding with sin cos setups-- consider alternate interleaving?
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return pos_encoding


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

def build_bbc_data(input_file='../data/v2/bbc_neutral.json'):

    with open(input_file) as f:
        data = json.load(f)

    # separate content and title data into separate lists
    bbc_test_content = [item['content'] for item in data]
    bbc_test_title = [item['title'] for item in data]
    return bbc_test_content, bbc_test_title


if __name__ == '__main__':
    train_content, train_title, test_content, test_title = train_test_split()
    bbc_content, bbc_title = build_bbc_data()
    (content_vocab, content_word_index, content_index_word,
     title_vocab, title_word_index, title_index_word) = vectorize_data(train_content, train_title)
    glove_index = build_glove_embed_index()

    train_content_emb = create_embeddings(train_content, glove_index, 256, 100, 'train_content')
    test_content_emb = create_embeddings(test_content, glove_index, 256, 100, 'test_content')
    bbc_content_emb = create_embeddings(bbc_content, glove_index, 256, 100, 'bbc_content')

    train_title_emb = create_embeddings(train_title, glove_index, 16, 100, 'train_title')
    test_title_emb = create_embeddings(test_title, glove_index, 16, 100, 'test_title')
    bbc_title_emb = create_embeddings(bbc_title, glove_index, 16, 100, 'bbc_title')

    train_title_labels = create_token_labels(train_title, TITLE_VECTORIZER, dataset_name='train')
    test_title_labels = create_token_labels(test_title, TITLE_VECTORIZER, dataset_name='test')
    bbc_title_labels = create_token_labels(bbc_title, TITLE_VECTORIZER, dataset_name='bbc')


    save_to_pickle(train_content_emb, '../data/embeddings/train_content_embeddings.pkl')
    save_to_pickle(test_content_emb, '../data/embeddings/test_content_embeddings.pkl')
    save_to_pickle(train_title_emb, '../data/embeddings/train_title_embeddings.pkl')
    save_to_pickle(test_title_emb, '../data/embeddings/test_title_embeddings.pkl')
    save_to_pickle(train_title_labels, '../data/embeddings/train_title_labels.pkl')
    save_to_pickle(test_title_labels, '../data/embeddings/test_title_labels.pkl')
    save_to_pickle(title_index_word, '../data/embeddings/title_index_word.pkl')

    save_to_pickle(bbc_content_emb, '../data/embeddings/bbc_content_embeddings.pkl')
    save_to_pickle(bbc_title_emb, '../data/embeddings/bbc_title_embeddings.pkl')
    save_to_pickle(bbc_title_labels, '../data/embeddings/bbc_title_labels.pkl')
