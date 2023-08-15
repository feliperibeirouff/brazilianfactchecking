from pathlib import Path
from datetime import datetime
import csv
import os
import dataset_loader
import json
import time

def isValid(url):
  invalid_extensions = ['.pdf', '.doc', '.docx', '.txt', '.zip']
  for i in invalid_extensions:
    if i in url.lower():
      return False
  return True      


class GoogleSearcher:
  def __init__(self, googleApi = None):
    try:
      with open("config.json", "r") as file:
        self.config = json.load(file)
    except:
      print("Error reading config.json")
      self.config = {'googleApi': 'googlesearch'}

    print('googleApi:', googleApi)
    if googleApi:
      self.config['googleApi'] = googleApi
    
    if self.config['googleApi'] == 'googlesearch':
      print("Googlesearch")
    else:
      print("Google api client")
      from googleapiclient.discovery import build
      self.service = build(
        "customsearch", "v1", developerKey=self.config["developerKey"]
      )

  def preprocessAndDoSearch(self, search_query, num_documents_by_claim, out_json_file_path, MAX_SIZE, before_date = None, identifier = 0, only_print_queries=False):
    if len(search_query) > MAX_SIZE:
      print(identifier, 'exceeds maximum search size', len(search_query))
      search_query = search_query[:MAX_SIZE]

    if before_date:
      date_string = before_date.strftime("%Y-%m-%d")
      search_query = "before:" + date_string + " " + search_query
    if only_print_queries:
      print('id:', identifier, 'query:', search_query)
      return []
    return self.doSearch(search_query, num_documents_by_claim, out_json_file_path)

  def doSearch(self, search_query, num_documents_by_claim, out_json_file_path = None):
    if self.config['googleApi'] == 'googlesearch':
      try:
        from googlesearch import search
      except ImportError: 
        print("No module named 'google' found")
      #print(search_query)
      urls = search(search_query, tld="com.br", num=num_documents_by_claim, stop=num_documents_by_claim, pause=5)
      urls = [url for url in urls] #A leitura só é efetuada ao iterar a lista
    else:
      res = (
          self.service.cse()
          .list(
            q=search_query,
            cx=self.config["cx"]
          )
          .execute()
        )
      with open(out_json_file_path, "w") as outjsonfile:
        json.dump(res, outjsonfile, indent = 2)
      if 'items' not in res:
        print("Not found:" + search_query)
        urls = []
      else:
        urls = [item['link'] for item in res['items'] if True or isValid(item['link'])]
        #urls = urls[:num_documents_by_claim]        
    return urls

def documentRetrieval(claims, documents_path, num_documents_by_claim = 5, MAX_SIZE = 300, days_before_date = 0, only_print_queries=False):
  """! Retrieves documents related to the claims and saves them to a file
  It is necessary to have a config.json file that informs which api will be used, having the following value options for googleApi:
  1-'googlesearch' = https://pypi.org/project/googlesearch-python/
  2-'googleDevelopersApiClient' = https://developers.google.com/custom-search/v1/introduction?hl=pt-br
  In the second option, it is necessary to follow the instructions on the website to create the API key and cx.
  They need to be informed in config.json. Ex.:
  {
    "googleApi":"googleDevelopersApiClient",
    "developerKey":"DFSjqereqwruudoasdufs",
    "cx":"f558dfa3dcec"
  }  
  If the config.json file does not exist, googlesearch will be used

  @param claims - List of claims. Each item needs to have {'id': unique identifier, 'claim_clean': pre-processed claim's text}
  @param documents_path - path where the file will be saved
  @param num_documents_by_claim - number of searched documents
  @param MAX_SIZE - maximum number of characters used in the query
  """
  searcher = GoogleSearcher()

  last_id = -1
  try:
    with Path(documents_path).open('r', encoding="utf-8") as f:
      try:
        c = f.readlines()[-1].split("\t")[0]
        last_id = int(c)
      except:
        pass
  except:
    print("Error reading document file.")
  print('last_id:', last_id)
    
  ignore_claims = (last_id != -1)

  service = None
  qtd = 0
  last_time = datetime.now()
  print('last_time', last_time)
  with Path(documents_path).open('a', encoding="utf-8", newline='') as f:
    fieldnames = ['claim_id', 'rank', 'found_url']
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
    if last_id == -1:
      writer.writeheader()
    print('size: ', len(claims))
    for i, claim in enumerate(claims):
      if ignore_claims or int(claim['id']) == last_id:
        if int(claim['id']) == last_id:
          ignore_claims = False
        #print('ignorando ', i, claim['id'])
        continue
      qtd += 1
      
      if searcher.config['googleApi'] != 'googlesearch' and qtd % 99 == 0:
        seconds = (datetime.now() - last_time).total_seconds()
        if seconds < 61: #Avoiding exceeding quota of 100 requests per minute
          print("wait seconds:", (61 - seconds))
          time.sleep(61 - seconds)
        last_time = datetime.now()
        print('last_time', last_time, qtd)

      if i % 10 == 0:
        print(str(datetime.now()), i, claim['id'])
      

      search_query = claim['claim_clean']

      before_date = None
      if days_before_date != 0:
        date_string = claim['date_published']
        before_date = datetime.strptime(date_string, '%Y/%m/%d') - timedelta(days=days_before_date)

      json_out_path = documents_path + "_" + str(claim['id']) + ".json"

      urls = searcher.preprocessAndDoSearch(search_query, num_documents_by_claim, json_out_path, MAX_SIZE, before_date, identifier=str(i) + " " + str(claim['id']), only_print_queries=only_print_queries)

      for j, url in enumerate(urls):        
        row = {'claim_id': claim['id'], 'rank': j, 'found_url': url}
        writer.writerow(row)   
      f.flush()

def getDocumentExtractedPath(urls_directory, document):
  return urls_directory + str(document['row']) + "_" + str(document['claim_id']) + "_" + str(document['rank']) + "_extracted.txt"

def extractDocuments(documents_path, urls_directory, url_categories_path):
  """! Opens the list of documents and, for each url, extracts the site and saves it to a file
  @param documents_path - Document file path. Must end with .tsv and have the fields {row: document row, claim_id: clam id, rank, found_url}
  """
  from url_extractor import UrlExtractor
  extractor = UrlExtractor()

  documents = dataset_loader.loadDocuments(documents_path, url_categories_path)
  os.makedirs(urls_directory, exist_ok=True) 
  logfile = open(urls_directory + "log.txt", "a", encoding="utf-8", newline='')
  fieldnames = ['row', 'claim_id', 'rank', 'found_url', 'qtd_lines', 'qtd_char', 'error']
  logwriter = csv.DictWriter(logfile, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
  logwriter.writeheader()
  for document in documents:
    document_path = getDocumentExtractedPath(urls_directory, document)
    extractor.save_document(document, document_path, logwriter, False)
    logfile.flush()
  logfile.close()