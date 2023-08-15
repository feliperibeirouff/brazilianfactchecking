import csv

from datetime import datetime

def standardizeDate(date):
  date = date.replace("-","/")
  try:
    d = datetime.strptime(date, '%d/%m/%y')
    date = d.strftime("%Y/%m/%d")
  except:
    try:
      d = datetime.strptime(date, '%d/%m/%Y')
      date = d.strftime("%Y/%m/%d")
    except:
      pass
  return date

def loadClaims(filename, delimiter="\t"):
  """! Load the list of claims
  @param filename - path of the file with the claims
  The file needs to have the fields {'id': unique identifier of the claim, 'claim_clean': the text of the claim (preferably already pre-processed)}
  @param delimiter - file delimiter
   """
  claims = []
  with open(filename,'r', encoding='utf-8') as f:
    read_claims = csv.DictReader(f, delimiter=delimiter, skipinitialspace=True)
    i = 0
    for claim in read_claims: 
      claim['row'] = i
      claim['id'] = int(claim['id'])
      claim['date_published'] = standardizeDate(claim['date_published'])
      claims.append(claim)
      i += 1
  return claims

def loadUrlCategories(url_categories_path):
  url_categories = []
  with open(url_categories_path,'r', encoding='utf-8') as f:
    read = csv.DictReader(f, delimiter="\t", skipinitialspace=True)
    for pattern in read:
      url_categories.append(pattern)
  return url_categories


def getCategory(url, url_categories):
  category = ''
  valid = False
  for s in url_categories:
    if s['site'] in url.lower():
      category = s['categoria']
      if 'ruim' not in category:
        valid = True
      break
  if valid:
    invalid_extensions = ['.pdf', '.doc', '.docx', '.txt', '.zip']
    for i in invalid_extensions:
      if i in url.lower():
        valid = False
        category = 'ruim (arquivo)'
        break
  return category

def loadDocuments(documents_path, url_categories_path):
  documents = []
  url_categories = loadUrlCategories(url_categories_path)
  with open(documents_path,'r', encoding='utf-8') as f:
    read_urls = csv.DictReader(f, delimiter="\t", skipinitialspace=True)
    i = 0
    for url in read_urls:    
      url['claim_id'] = int(url['claim_id'])
      url['rank'] = int(url['rank'])
      url['category'] = getCategory(url['found_url'], url_categories)
      url['row'] = i
      documents.append(url)
      i += 1
  return documents

def loadEvidences(filename):
  trechos = []
  with open(filename,'r', encoding='utf-8') as f:
    read_trechos = csv.DictReader(f, delimiter="\t", skipinitialspace=True)
    i = 0
    for trecho in read_trechos: 
      trecho['row'] = i
      trecho['claim_row'] = int(trecho['claim_row'])
      trecho['claim_id'] = int(trecho['claim_id'])
      trecho['similarity'] = float(trecho['similarity'])
      trecho['evidence_row'] = int(trecho['evidence_row'])
      trechos.append(trecho)
      i += 1
  return trechos