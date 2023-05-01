import numpy as np
import tensorflow as tf
import pickle
import json

np.random.seed(2470)

def train_test_split(input_file='../data/nytfox_collate_v2.json', test_split=0.2):
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
    np.random.shuffle(idx)

    # create train and test sets for content and titles
    temp_content_arr = np.array(content_arr)[idx]
    temp_title_arr = np.array(title_arr)[idx]

    train_content = (temp_content_arr.tolist())[:-num_test_samples]
    test_content = (temp_content_arr.tolist())[-num_test_samples:]

    train_title = (temp_title_arr.tolist())[:-num_test_samples]
    test_title = (temp_title_arr.tolist())[-num_test_samples:]

    return train_content, train_title, test_content, test_title


if __name__ == '__main__':
    train_content, train_title, test_content, test_title = train_test_split()



