from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import json
import datetime as dt
import time
import os


def remove_older_versions_biorxiv():
    dir = 'biorxivTDM'
    numbers = {}
    for file in os.listdir(dir):
        number = file.split('v')
        version = number[1].split('.')[0]
        if number[0] not in numbers:
            numbers[number[0]] = (file, version)  # saves filename and version
        else:
            if int(version) > int(numbers[number[0]][1]):
                path = os.path.join(dir, numbers[number[0]][0])
                os.remove(path)
                numbers[number[0]] = (file, version)
            else:
                path = os.path.join(dir, file)
                os.remove(path)


def download_data():
    options = Options()
    options.add_argument('--disable-extensions')
    options.add_argument(
        'user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36')
    driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver", options=options)
    date = dt.datetime.strptime('2020-12-01', '%Y-%m-%d')

    options = Options()
    options.add_argument(
        'user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36')
    options.add_argument(
        'accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9')
    options.add_argument('accept-encoding=gzip, deflate, br')
    options.add_argument('accept-language=en-US,en;q=0.9')
    options.add_experimental_option('prefs', {
        "download.default_directory": "/home/marija/PycharmProjects/data/biorxivTDM",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
    })
    driver1 = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver", options=options)
    meta_data = []
    urls = []
    first_save = True

    while date.date() != date.now().date():
        next_date = date + dt.timedelta(days=1)
        collection_len = 100
        cnt = 1
        while collection_len == 100:
            driver.get(f"https://api.biorxiv.org/details/biorxiv/{date.date()}/{next_date.date()}/{cnt}")
            json_string = driver.page_source.split('>')[5].split('<')[0].strip()
            data = json.loads(json_string)["collection"]
            if len(data) > 0:
                for pub in data:
                    if pub['doi'] not in urls:
                        driver1.get(f"https://www.biorxiv.org/content/{pub['doi']}v{pub['version']}.full.pdf")
                        time.sleep(1)
                        pub['title'] = '\"' + pub['title'] + '\"'
                        pub['authors'] = '\"' + pub['authors'] + '\"'
                        meta_data.append([pub['doi'], pub['version'], pub['title'], pub['authors'],
                                          pub['date'], pub['category'], pub['published']])
                        urls.append(pub['doi'])
            collection_len = len(data)
            cnt += 100
        if len(meta_data) > 1000:
            metadata = pd.DataFrame(meta_data)
            metadata.columns = ['ID', 'Version', 'Title', 'Authors', 'Date', 'Category', 'Published']
            metadata.to_csv('biorxivTDM/biorxivTDM_metadata.csv', mode='a', quotechar='"', header=first_save,
                            index=False)
            first_save = False
            meta_data = []
            urls = []
        date = next_date


os.chdir('data')
download_data()
# Authors can reupload update in biorxiv. This function deletes older versions from downloaded papers
remove_older_versions_biorxiv()
