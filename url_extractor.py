from selenium.common.exceptions import NoSuchElementException
import preprocessing
from pathlib import Path
import csv
import re
from urllib.request import urlopen
from urllib.request import Request
from bs4 import BeautifulSoup
import ssl
import os
import time

class UrlExtractor:  
  def __init__(self):
    self.driver = None


  def getDriver(self):
    if not self.driver:
      from selenium import webdriver
      from selenium.webdriver.common.by import By
      from selenium.webdriver.support.ui import WebDriverWait
      from selenium.webdriver.support import expected_conditions as EC
      print("inicializando driver do selenium")
      self.driver = webdriver.Firefox()
    return self.driver
    

  def getSeleniumContent(self, url):
    driver = self.getDriver()
    driver.get(url)    
    title = driver.title
    content = driver.find_element_by_tag_name('body').text
    text = content
    text = re.sub(r'Leia também.*','', text, flags=re.S)
    return title + "\n" + text

  def getLupaContent(self, url):    
    driver = self.getDriver()
    driver.get(url)   
    time.sleep(5)
    title = driver.title
    content = driver.find_element_by_tag_name('body').text
    text = content
    text = re.sub(r'Nota: esta reportagem faz parte do projeto de verificação.*','', text, flags=re.S)
    text = re.sub(r'A Lupa faz parte do.*','', text, flags=re.S)
    index = content.find('Entrar')
    if index>0:
        text = text[index+len('Entrar'):]
    return title + "\n" + text

  def extract_url(self, url):
      error = ""
      try:
          req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
          response = urlopen(req, timeout=10)
          contentType = response.headers['Content-Type']
          if 'html' not in contentType:
              return [], 'Content type ' + contentType
          html = response.read()
      except:
          try:
              gcontext = ssl.SSLContext()
              response = urlopen(url, timeout=10, context=gcontext)
              contentType = response.headers['Content-Type']
              if 'html' not in contentType:
                  return [], 'Content type ' + contentType
              html = response.read()
          except Exception as e:
              error = "Erro lendo url " + repr(e)
              print("ERRO " + error)
              return [], error
      #print(len(html))
      #html = urlopen(url).read()
      
      MAX_LENGTH = 1000000
      if len(html) > MAX_LENGTH:
          soup = BeautifulSoup(html[:MAX_LENGTH], features="html.parser")
      else:
          soup = BeautifulSoup(html, features="html.parser")
      # kill all script and style elements
      for script in soup(["script", "style"]):
          script.extract()    # rip it out

      # get text
      text = soup.get_text()       
          
      extracted_lines = []

      # break into lines and remove leading and trailing space on each
      lines = (line.strip() for line in text.splitlines())
      # break multi-headlines into a line each
      chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
      # drop blank lines
      for chunk in chunks:
          if len(chunk) > 10000:
              error = "len(chunk) " + str(len(chunk))
              print("ERRO " + error)
          elif len(chunk) > 30:
              # print('len_chunk', len(chunk))
              extracted_lines.append(chunk) 
              #print(chunk)
      return extracted_lines, error

  def save_document(self, document, document_path, logwriter, replace): 
    if document['category'] != '' and 'ruim' not in document['category']:    
      if not os.path.exists(document_path) or replace:
        print(document['row'], document['claim_id'], document['rank'], document['found_url'], document['category'])
        url = document['found_url']
        if "lupa" in url:
          extracted_lines = [self.getLupaContent(url)]
          error = ""
        if "exame.com" in url:
          extracted_lines = [self.getSeleniumContent(url)]
          error = ""
        else:
          extracted_lines, error = self.extract_url(url)
          if error == 'Erro lendo url' in error:
            extracted_lines = [self.getSeleniumContent(url)]
            error = ""
        clean_lines = []
        for line in extracted_lines:
          clean_line = preprocessing.tokenize_and_join(line)
          if clean_line != '':
            clean_lines.append(clean_line)        
        #print('clean', clean_lines)

        document_raw_path = document_path + '_raw.txt'
        f_raw = open(document_raw_path, "w", encoding="utf-8")
        lines_join_raw = "\n".join(extracted_lines)        
        f_raw.write(lines_join_raw)
        f_raw.close()

        f = open(document_path, "w", encoding="utf-8")
        lines_join = "\n".join(clean_lines)
        print(lines_join.count('\n'), len(lines_join))
        f.write(lines_join)
        f.close()    
        logwriter.writerow({'row': document['row'], 'claim_id': document['claim_id'],
          'rank': document['rank'], 'found_url': url, 'qtd_lines': lines_join.count('\n'), 'qtd_char' : len(lines_join), 'error': error})