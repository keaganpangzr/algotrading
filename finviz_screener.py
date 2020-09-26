import csv
import requests
from bs4 import BeautifulSoup
import math
import json

headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-GB; rv:1.8.1.6) Gecko/20070725 Firefox/2.0.0.6'}



#copy and paste first part of URL from browser
URL = 'https://www.finviz.com/screener.ashx?v=111&f=cap_midover,fa_div_none,sh_avgvol_o200,sh_price_10to50' + '&r='
page = requests.get(URL, headers = headers)
soup = BeautifulSoup(page.content, 'html.parser')

def scrape_all_tickers(URL, total_tickers):
    tickers = []

    page_count = math.ceil(total_tickers / 20)
    ticker_count = 0

    for page_count in range(page_count):
        
        page = requests.get(URL + str(ticker_count), headers = headers)
        soup = BeautifulSoup(page.content, 'html.parser')   
        result = soup.find_all('a', attrs={'class':'screener-link-primary'})

        for i in range(len(result)):
            for j in result[i]:
                tickers.append(j)

        ticker_count += 20 

    print(f'{len(tickers)} ticker symbols scraped')
    print(tickers)
    return tickers

def write_to_json(tickers):
    """Write list of tickers to json file, saved in specified dir"""

    with open('finviz_screen1/screen1.json','w') as f:
        json.dump(tickers, f)

write_to_json(scrape_all_tickers(URL, 289))









