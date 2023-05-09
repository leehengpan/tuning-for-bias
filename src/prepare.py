import re
from urllib.parse import unquote, unquote_plus
import json
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


def collate_data(files, save_to=None, clean_text=None, article_limit='all',
                 set_views='from_file', map_file=None):
    """Collate data from news article files and return in a consistent format. Save to json file, if necessary.
    Argyments:
        files = array of filepaths (expects jsons)
        set_views = 'from_file' or [Str]; 'from_file' finds the view from the field 'view' in the JSON file, else pass list of custom strings
        save_to = filepath of the destination json file; optional
        clean_text = list of functions to remove special characters, encoding, stemming, etc.
        map_file = Maps titles to index and view.
        article_limit =  define custom limit to process number of articles from each file. 'all' by default."""

    assert set_views=='from_file' or isinstance(set_views, list), "set_views must be 'from_file' or custom values from a list"

    result = []
    view_maps = []

    for file in files:
        with open(file) as f:
            info_arr = json.load(f)
            for i, info_dict in enumerate(info_arr):
                if i==article_limit:
                    break
                temp_dict = {}
                if info_dict is None:
                    continue
                try:
                    if (info_dict['content'] == '' or info_dict['content'] is None or
                            info_dict['title'] == '' or info_dict['title'] is None):
                        continue

                    if set_views=='from_file':
                        map_index = i
                        views = [info_dict['view']]
                    else:
                        views = set_views
                        map_index = i*len(views)

                    for j,view in enumerate(views):
                        view_map = {}
                        temp_dict['content'] = info_dict['content']
                        temp_dict['title'] = info_dict['title']

                        if clean_text:
                            for func in clean_text:
                                temp_dict['content'] = func(temp_dict['content'])
                                temp_dict['title'] = func(temp_dict['title'])

                        temp_dict['content'] = view + ' : ' + temp_dict['content']
                        view_map['index'] = map_index+j
                        view_map['view'] = view
                        view_map['original_title'] = temp_dict['title']
                        view_map['original_view'] = info_dict.get('view','NA')

                        result.append(temp_dict)
                        view_maps.append(view_map)

                except Exception as e:
                    print(i, e)
            print(f'Processed {i} articles in file {file}')

    if save_to:
        with open(save_to, 'w+', encoding='utf-8') as f:
            json.dump(result, f)
            print(f'Saved to {save_to}')

    if map_file:
        df = pd.DataFrame(view_maps)
        df['index'] = df.index
        df.to_csv(map_file, index=False, encoding='utf-8')
        print(f'Saved map to {map_file}')

if __name__ == '__main__':
    files = ['../data/foxnews_content.json', '../data/nyt_content.json']
    clean_text = [remove_char_encoding, remove_special_char, make_lowercase]
    collate_data(files, save_to='../data/nytfox_collate.json', clean_text=clean_text)




