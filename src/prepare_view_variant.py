import re
from urllib.parse import unquote, unquote_plus
import json
from preprocess import *
import pandas as pd


def remove_special_char(txt):
    """Remove special characters from text."""

    special_char = re.compile(r'[^A-Za-z0-9\s\-\/]+', re.IGNORECASE)
    special_char_space = re.compile(r'[\-\/]+', re.IGNORECASE)
    txt = str(txt).lower().strip()
    check_list = special_char.findall(txt)
    for item in check_list:
        try:
            txt = txt.replace(item, '')

        except Exception as e:
            print(e)

    check_list_space = special_char_space.findall(txt)
    for item in check_list_space:
        try:
            txt = txt.replace(item, ' ')
        except Exception as e:
            print(e)

    return txt.strip()


def make_lowercase(txt):
    return txt.lower()


def remove_char_encoding(txt):
    """Remove character encoding from text."""
    txt = str(txt)
    return unquote(unquote_plus(txt.encode('ascii', 'ignore').decode()))


def collate_data(files, save_to=None, clean_text=None, mapfile='../data/v2/nytfox_view_based_map.csv'):
    """Collate data from news article files and return in a consistent format. Save to json file, if necessary.
    Argyments:
        files = array of filepaths (expects jsons)
        save_to = filepath of the destination json file; optional
        clean_text = list of functions to remove special characters, encoding, stemming, etc."""

    result = []
    view_maps = []

    for file in files:
        with open(file) as f:
            info_arr = json.load(f)
            for i, info_dict in enumerate(info_arr):
                if (i % 1000 == 0):
                    print(f'Processed {i} articles in file {file}')
                for j,view in enumerate(['liberal', 'conservative']):
                    view_map = {}
                    temp_dict = {}
                    if info_dict is None:
                        continue
                    try:
                        if (info_dict['content'] == '' or info_dict['content'] is None or
                                info_dict['title'] == '' or info_dict['title'] is None):
                            continue

                        temp_dict['content'] = info_dict['content']
                        temp_dict['title'] = info_dict['title']

                        if clean_text:
                            for func in clean_text:
                                temp_dict['content'] = func(temp_dict['content'])
                                temp_dict['title'] = func(temp_dict['title'])

                        temp_dict['content'] = view + ' : ' + temp_dict['content']
                        view_map['index'] = i+(i+j)
                        view_map['view'] = view
                        view_map['original_title'] = temp_dict['title']
                        view_map['original_view'] = info_dict['view']

                        result.append(temp_dict)
                        view_maps.append(view_map)

                    except Exception as e:
                        print(i, e)
    if save_to:
        with open(save_to, 'w+', encoding='utf-8') as f:
            json.dump(result, f)
            print(f'Saved to {save_to}')

    if mapfile:
        df = pd.DataFrame(view_maps)
        df.to_csv(mapfile, index=False, encoding='utf-8')

if __name__ == '__main__':
    files = ['../data/v2/foxnews_content_v2.json', '../data/v2/nyt_content_isaac.json',
             '../data/v2/nyt_content_liam.json', '../data/v2/nyt_content_sagar.json']
    clean_text = [remove_char_encoding, remove_special_char, make_lowercase]
    filepath = '../data/v2/nytfox_bias_collate.json'
    collate_data(files, save_to=filepath, clean_text=clean_text)

    train_content, train_title, test_content, test_title = train_test_split(input_file=filepath, test_split=0.01, shuffle=False)
    (content_vocab, content_word_index, content_index_word,
     title_vocab, title_word_index, title_index_word) = vectorize_data(train_content, train_title)
    glove_index = build_glove_embed_index()

    train_content_emb = create_embeddings(train_content, glove_index, 256, 100, 'train_content')
    train_title_emb = create_embeddings(train_title, glove_index, 16, 100, 'train_title')
    train_title_labels = create_token_labels(train_title, TITLE_VECTORIZER, dataset_name='train')

    save_to_pickle(train_content_emb, '../data/embeddings/bias_content_embeddings.pkl')
    save_to_pickle(train_title_emb, '../data/embeddings/bias_title_embeddings.pkl')
    save_to_pickle(train_title_labels, '../data/embeddings/bias_title_labels.pkl')
    save_to_pickle(title_index_word, '../data/embeddings/bias_index_word.pkl')