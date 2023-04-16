from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, NoSuchElementException
import time
import numpy as np
import pandas as pd

import csv
import json
from newspaper import Article
from newspaper.article import ArticleException

browser = webdriver.Chrome(desired_capabilities={"page_load_strategy": "none"})
browser.set_page_load_timeout(5)
browser.maximize_window()


def fetch_category_page(page, show_more_click=0, show_more_xpath=None, articles_xpath=None):
    print(page)
    try:
        browser.get(page)
    except TimeoutException:
        print('Loaded for 5 secs... moving on')

    time.sleep(np.max([0, np.random.normal(loc=5, scale=0.5)]))

    for _ in range(show_more_click):
        if _ % 50 == 0:
            print(_)
        time.sleep(1)

        try:
            load_more_element = browser.find_element(By.XPATH, show_more_xpath)
            load_more_element.click()

        except NoSuchElementException as e:
            print(e)
            break

    article_elements = browser.find_elements(By.XPATH, articles_xpath)
    articles = []

    for i, art in enumerate(article_elements):
        try:
            info_arr = art.find_element(By.CLASS_NAME, 'info')
            info_header = info_arr.find_element(By.CLASS_NAME, 'info-header')
            tag = info_header.find_element(By.CLASS_NAME, 'meta').find_element(By.CLASS_NAME, 'eyebrow').text
            title = info_header.find_element(By.CLASS_NAME, 'title').text
            link = info_arr.find_element(By.CLASS_NAME, 'title').find_element(By.XPATH, 'a').get_attribute('href')
            desc = info_arr.find_element(By.CLASS_NAME, 'content').text

            if i % 100 == 0:
                print(i)

            articles.append([title, link, tag, desc])

        except Exception as e:
            print(i, e)
            print(info_arr.text)

    df = pd.DataFrame(articles, columns=['title', 'link', 'tag', 'desc'])
    return df


if __name__ == '__main__':
    pull_map_from = '../data/fox-news-link-xpath-map.csv'
    save_map_to = '../data/fox-news-map.csv'

    concat_df = pd.DataFrame()

    with open(pull_map_from) as pull_map:
        page_info = csv.DictReader(pull_map)

        concat_df = pd.DataFrame()

        for info in page_info:
            page_info_df = fetch_category_page(info['url'], int(info['show_more']), info['footer'], info['article'])
            page_info_df['page'] = info['url']
            page_info_df['topic'] = info['topic']
            concat_df = pd.concat([concat_df, page_info_df])

    print(concat_df.shape)
    concat_df.drop_duplicates().to_csv(save_map_to, index=False, mode='a')
