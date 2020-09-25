import csv
import requests
from bs4 import BeautifulSoup
import re
import math

headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-GB; rv:1.8.1.6) Gecko/20070725 Firefox/2.0.0.6'}



#copy and paste first part of URL from browser
URL = 'https://www.finviz.com/screener.ashx?v=111&f=cap_midover,fa_div_none,geo_usa,ind_airlines' + '&r='
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

    print(tickers)
    print(len(tickers))

#remember to include total no. of expected results so function knows how many pages to scrape (20 results per page)
scrape_all_tickers(URL, 6)









