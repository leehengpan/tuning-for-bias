import re
from urllib.parse import unquote, unquote_plus
import json

def remove_special_char(txt):
    """Remove special characters from text."""
    FIRST_LAST_SC = re.compile(r'[^A-Za-z0-9\s]+', re.IGNORECASE)
    txt = str(txt).lower().strip()
    check_list = FIRST_LAST_SC.findall(txt)
    for item in check_list:
        try:
            txt = txt.replace(item, '')
        except Exception as e:
            print(e)
    return txt.strip()

def make_lowercase(txt):
    return txt.lower()


def remove_char_encoding(txt):
    """Remove character encoding from text."""
    txt = str(txt)
    return unquote(unquote_plus(txt.encode('ascii', 'ignore').decode()))


def collate_data(files, save_to=None, clean_text=None):
    """Collate data from news article files and return in a consistent format. Save to json file, if necessary.
    Argyments:
        files = array of filepaths (expects jsons)
        save_to = filepath of the destination json file; optional
        clean_text = list of functions to remove special characters, encoding, stemming, etc."""

    result = []

    for file in files:
        with open(file) as f:
            info_arr = json.load(f)
            for i, info_dict in enumerate(info_arr):
                if (i % 1000==0):
                    print(f'Processed {i} articles in file {file}')
                temp_dict = {}
                try:
                    if info_dict['content'] == '' or info_dict['content'] is None or \
                            info_dict['title'] == '' or info_dict['title'] is None:
                        continue

                    temp_dict['content'] = info_dict['content']
                    temp_dict['title'] = info_dict['title']
                    if clean_text:
                        for func in clean_text:
                            temp_dict['content'] = func(temp_dict['content'])
                            temp_dict['title'] = func(temp_dict['title'])
                    result.append(temp_dict)

                except Exception as e:
                    print(i, e)
    if save_to:
        with open(save_to, 'w+', encoding='utf-8') as f:
            json.dump(result, f)
            print(f'Saved to {save_to}')


files = ['../data/foxnews_collate_content_v1.json', '../data/nyt_collate_content_v1.json']
clean_text = [remove_char_encoding, remove_special_char, make_lowercase]
collate_data(files, save_to='../data/nytfox_collate_v2.json', clean_text=clean_text)
