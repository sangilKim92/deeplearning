from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import datetime
import codecs
import webbrowser
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
from bs4 import BeautifulSoup
from konlpy.tag import Okt
options = Options()
options.headless = True  # //// 이 구문을 실행시키면 window 화면에 실행 없이 Background에서 실행된다.
browser = webdriver.Chrome('C:/Users/user/Downloads/chromedriver_win32/chromedriver')
# 직접 실행시키고
browser.implicitly_wait(0.2)  # 1초 휴식

url = 'https://play.google.com/store/apps/details?id=com.hanaskcard.paycla&showAllReviews=true'  # +page_num+str(a)
print(url)
browser.get(url)
time.sleep(0.5)

#최신목록으로 바꾸는 코드
browser.find_element_by_xpath('//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div/main/div/div[1]/div[2]/c-wiz/div[1]/div/div[1]/div[2]/span').click()

browser.find_element_by_xpath('//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div/main/div/div[1]/div[2]/c-wiz/div[1]/div/div[2]/div[1]/span').click()

#스크롤 내리는 코드
last_page_height = browser.execute_script("return document.documentElement.scrollHeight")
for b in range(2):
    for a in range(10):
        browser.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(3.0)
        new_page_height = browser.execute_script("return document.documentElement.scrollHeight")
        last_page_height = new_page_height
    time.sleep(5.0)
    #더 보기 클릭하는 코드
    browser.find_element_by_xpath('//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div/main/div/div[1]/div[2]/div').click()


#fcxH9b > div.WpDbMd > c-wiz > div > div.ZfcPIb > div > div > main > div > div.W4P4ne > div:nth-child(2) > div > div:nth-child(1) > div > div.d15Mdf.bAhLNe > div.UD7Dzf > span:nth-child(1)

#내용 가져오는 select문
soup = BeautifulSoup(browser.page_source, 'html.parser')
ss=soup.select('div.W4P4ne > div > div > div > div > div > div > span')

print(ss[0].text)

#별점 가져오는 select문 list를 개수세어 해결해야 할듯
ss=soup.select('div.W4P4ne > div > div > div > div > div > div > div > div .pf5lIe > div > div ')
print(ss[0])
ss[0]['class'][0]