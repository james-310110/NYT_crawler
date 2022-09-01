# from dataclasses import field
# from email.mime import base
# from itertools import count
# from pydoc import describe
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, NoSuchAttributeException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cchardet
from datetime import date, timedelta
import pandas as pd
import requests
import time
import csv


############ COLLECT URLS USING SELENIUM ############

def extract_urls(base_url,driver):
    driver.get(base_url)
    # wait = WebDriverWait(driver,1)
    while True:
        time.sleep(0.4)
        for _ in range(5):
            actions = ActionChains(driver)
            actions.send_keys(Keys.END).perform()
            time.sleep(0.02)
        # using explicit wait time to prevent unwanted clicking
        time.sleep(0.5)
        try:
            button = driver.find_element(By.XPATH,'//button[@data-testid="search-show-more-button" and text()="Show More"]')
        except NoSuchElementException:
            break
        else:
            button.click()

    links = driver.find_elements(By.TAG_NAME,'a')
    page_urls = []
    counter = 0
    for link in links:
        try:    
            link_url = link.get_attribute('href')
        except NoSuchAttributeException:
            pass
        else:
            if link_url is not None:
                if 'https://www.nytimes.com/20' in link_url:
                    counter += 1
                    page_urls.append(link_url)
    return page_urls, counter

############ SCRAPE DATA FROM EACH PAGE ############
    
def _extract_headline(soup):
    if soup.title is not None:
        return soup.title.get_text()[:-21]

def _extract_section(soup):
    section_element = soup.find('meta', property='article:section')
    if section_element is not None:
        return section_element['content']

def _extract_date(soup):
    date_element = soup.find('meta', property='article:published_time')
    if date_element is not None:
        return date_element['content'][:10]

def _extract_description(soup):
    description_element = soup.find('meta', {'name':'description'})
    if description_element is not None:
        return description_element['content']

def _extract_tags(soup):
    tags_element = soup.find_all('meta',property='article:tag')
    if tags_element is not None:
        return [tag['content'] for tag in tags_element]

def _extract_content(soup):
    content_element = soup.find_all('section', attrs={'name':'articleBody'})
    if content_element is not None:
        return ''.join([paragraph.get_text() for paragraph in content_element])

def extract_information(url, session, header):
    page = session.get(url, headers=header)
    soup = BeautifulSoup(page.text, 'lxml')
    try:
        headline = _extract_headline(soup)
        section = _extract_section(soup)
        date = _extract_date(soup)
        description = _extract_description(soup)
        tags = _extract_tags(soup)
        content = _extract_content(soup)
    except Exception as err:
        return None
    return {
        'date': date,
        'headline': headline,
        'section': section,
        'description': description,
        'tags': tags,
        'content': content,
        'url': url
    }

def crawl_urls(file_path):
    header =  {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15'}    
    csv_file = 'data/mega' #.csv'
    fields = ['date','headline','section','description','tags','content','url']
    page_urls = []
    infos = []
    counter = 25000
    session = requests.Session()
    with open(file_path,'r') as f:
        page_urls = f.readlines()[25000:]
    for url in page_urls:
        infos.append(extract_information(url,session,header))
        counter += 1
        # saving output every 1000 steps
        if counter % 20 == 0:
            print(counter, '...')
        if counter % 1000 == 0:
            with open(csv_file+str(counter/1000)+'.csv','w') as f:
                writer = csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(infos)
            print(counter, 'articles crawled from urls')
    with open(csv_file+'.csv','w') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(infos)
    print(counter, 'articles crawled from urls')

def collect_urls(start_year,end_year):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    dates = pd.date_range(
            str(start_year)+'-11-01',
            str(end_year)+'-01-01',
            freq='MS').strftime('%Y%m%d').tolist()
    all_urls = []
    total_count = 0
    for i in range(len(dates)-1):
        start_date = dates[i]
        end_date = dates[i+1]
        base_url = 'https://www.nytimes.com/search?dropmab=true&endDate='+end_date+'&query=china&sort=best&startDate='+start_date+'&types=article'
        page_urls, counter = extract_urls(base_url,driver)
        all_urls += page_urls
        total_count += counter
        
        # testing
        with open('data/page_urls_'+start_date[:6]+'.txt','w') as f:
            for url in all_urls:
                f.write('%s\n' % url)
        print(counter, 'urls collected from', start_date[:6])
        
    
def compile_urls(start_year,end_year):
    dates = pd.date_range(
            str(start_year)+'-11-01',
            str(end_year)+'-01-01',
            freq='MS').strftime('%Y%m%d').tolist()
    all_urls = []
    total_count = 0
    for i in range(len(dates)-1):
        start_date = dates[i]
        with open('data/page_urls_'+start_date[:6]+'.txt','r') as f:
            all_urls += f.readlines()
            print('urls collected from', start_date[:6])
    all_urls = list(set(all_urls))
    with open('data/page_urls_mega.txt','w') as f:
        for url in all_urls:
            f.write('%s' % url)
            total_count += 1
    print(total_count, 'urls collected from year',start_year,'to',end_year)


# collect_urls(2011,2021)
# compile_urls(2011,2021)
crawl_urls('data/page_urls_mega.txt')
    