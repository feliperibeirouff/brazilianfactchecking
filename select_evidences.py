import torch
import csv
from sentence_transformers import SentenceTransformer
import os
from scipy import spatial
from strategies import EvidenceSelectStrategy
import copy

def cos_similarity(sent1_emb, sent2_emb):
  return 1 - spatial.distance.cosine(sent1_emb, sent2_emb)

def document_extrated_path(document, directory):
  return directory + str(document['row']) + "_" + str(document['claim_id']) + "_" + str(document['rank']) + "_extracted.txt"

def document_sbert_path(document, directory):
  return document_extrated_path(document, directory).replace(".txt", "_sbert1.pt")

def documents_json_path(claim, directory):
   return directory.replace("_urls", "_json") + str(claim['id']) + ".json"

def sortFunction(e):
  return -(e['doc_has_quotes'] + e['value'])

class SbertEncoder:
  def __init__(self, urls_directory):
    self.urls_directory = urls_directory
    device = 'cuda'
    self.modelsbert = SentenceTransformer('models/portuguese_sentence_transformer')
    self.modelsbert.to(device)

  def sbert_sentence_embedding(self, claim):
    sentences = claim.split('\n')
    encoded = self.modelsbert.encode(sentences)
    return torch.mean(torch.Tensor(encoded), dim=0)

  def save_sbert_document(self, document_path, sbert_path):    
    if not os.path.exists(sbert_path):
      print(document_path)
      f = open(document_path, "r", encoding="utf-8")
      lines = f.readlines()  
      lines_sbert = []
      for line in lines:
        line_sbert = self.sbert_sentence_embedding(line.strip())
        lines_sbert.append(line_sbert)    
      torch.save(lines_sbert, sbert_path)

  def save_sbert_documents(self, documents):
    size = len(documents)
    for i, document in enumerate(documents):
      if document['category'] != '' and 'ruim' not in document['category']:
        print(i, size)
        document_path = document_extrated_path(document, self.urls_directory)
        sbert_path = document_sbert_path(document, self.urls_directory)
        self.save_sbert_document(document_path, sbert_path)

  def save_sbert_claims(self, claims, claims_sbert_path):
    claims_sbert = []
    print(len(claims))
    for i, claim in enumerate(claims):
      if i % 100 == 0:
        print(i)
      claim_sbert = self.sbert_sentence_embedding(claim['claim_clean'])
      claims_sbert.append(claim_sbert)
    torch.save(claims_sbert, claims_sbert_path)    


class EvidenceSelector:
  def __init__(self, urls_directory, include_title = False, remove_quotes = False, only_title = False):
    self.urls_directory = urls_directory
    self.include_title = include_title
    self.remove_quotes = remove_quotes
    self.only_title = only_title

  def retrieveDocuments(self, claim, google_results, maxDocuments = 5):
    claimid = claim['id']
    results_claim = []
    for result in google_results:
        if result['claim_id'] == claimid:
            results_claim.append(result)
    if len(results_claim) > maxDocuments:
        results_claim = results_claim[:maxDocuments]
    return results_claim

  def calculateSimilarities(self, claim_sbert, document):
    doc_sbert_path = document_sbert_path(document, self.urls_directory)
    if not os.path.exists(doc_sbert_path):
        return []
    document_sbert = torch.load(doc_sbert_path)
    document_path = document_extrated_path(document, self.urls_directory)
    f = open(document_path, "r", encoding="utf-8")
    document_lines = f.readlines()  
    document_lines = [row.replace('\0', '') for row in document_lines]
    similarities = []
    LIMIT_QUOTE = 0.95
    document_has_quotes = False
    for row, sentence_sbert in enumerate(document_sbert):
        #print('row:', row, 'line:', document_lines[row])
        quote = False
        similarity = cos_similarity(claim_sbert, sentence_sbert)    
        if similarity > LIMIT_QUOTE:
            quote = True
            document_has_quotes = True
        evidence = {'url': document['found_url'], 'row': row, 'text': document_lines[row].strip(),
          'value': similarity, 'quote': quote, 'doc_has_quotes': 0, 'doc_title':document_lines[0].strip()}
        similarities.append(evidence)
        if self.only_title: #Return only the first line, that has the title
           break
    if document_has_quotes:
        for s in similarities:
            s['doc_has_quotes'] = 1
    return similarities

  def retrieveEvidences(self, claim, claim_sbert, retrieved_documents):    
    #Calcula similaridades 
    similarities = []
    for document in retrieved_documents:
      similarities += self.calculateSimilarities(claim_sbert, document)
    similarities.sort(key=sortFunction)

    for similarity in similarities:
        for result in retrieved_documents:
            if result['found_url'] == similarity['url']:
                similarity['rank'] = result['rank'] 
    if self.remove_quotes:
      similarities = [s for s in similarities if not s['quote']]
    return similarities

  def selectEvidences(self, claims, documents, claims_sbert_path, evidences_path):
    claims_sbert = torch.load(claims_sbert_path)
    print(len(claims))
    with open(evidences_path,'w', encoding='utf-8', newline="") as f:
      fieldnames = ['claim_row', 'claim_id', 'similarity', 'claim_clean', 'evidence_clean', 'evidence_row', 'doc_rank', 'quote', 'doc_has_quotes', 'found_url', 'doc_title']
      writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
      writer.writeheader()
      for claim_row, claim in enumerate(claims):    
        if claim_row % 10 == 0:
            print(claim_row)
        claim_sbert = claims_sbert[claim_row]
        retrieved_documents = self.retrieveDocuments(claim, documents)
        evidences = self.retrieveEvidences(claim, claim_sbert, retrieved_documents)[:5]
        for evidence in evidences:
            if self.include_title and not evidence['text'].startswith(evidence['doc_title']):
              evidence['text'] = evidence['doc_title'] + ".\n " + evidence['text']
            #print(evidence)
            d = {'claim_row': claim_row, 'claim_id': claim['id'], 'claim_clean': claim['claim_clean'], 'found_url': evidence['url'], 
            'evidence_clean': evidence['text'], 'evidence_row': evidence['row'], 'doc_rank': evidence['rank'],
            'quote': evidence['quote'], 'doc_has_quotes': evidence['doc_has_quotes'], 'similarity':evidence['value'], 'doc_title': evidence['doc_title']}
            writer.writerow(d)

class EvidenceSelectorWithContext(EvidenceSelector):
  def __init__(self, urls_directory, context_sentences = 5, include_title = False, remove_quotes = None):
    super(EvidenceSelectorWithContext, self).__init__(urls_directory, include_title)    
    self.context_sentences = context_sentences
    self.remove_quotes = remove_quotes

  def retrieveEvidences(self, claim, claim_sbert, retrieved_documents):
    #Calculate similarities
    similarities_by_doc = {}
    similarities = []
    for document in retrieved_documents:
        doc_similarities = self.calculateSimilarities(claim_sbert, document)
        similarities += doc_similarities
        similarities_by_doc[document['rank']] = doc_similarities
    similarities.sort(key=sortFunction)

    #ensure that changes made to the text of the sentence will not alter the context of the other sentences
    similarities_by_doc = copy.deepcopy(similarities_by_doc)

    for similarity in similarities:
        for result in retrieved_documents:
            if result['found_url'] == similarity['url']:
                similarity['rank'] = result['rank']        

    for s in similarities[:5]:
      doc_sentences = similarities_by_doc[s['rank']]
      doc_sentences_filtered = doc_sentences[s['row'] + 1: s['row'] + self.context_sentences]
      if self.remove_quotes and s['quote']:
        context = '\n'.join([s['text'] for s in doc_sentences_filtered])
      else:
        context = s['text'] + ' . ' + '\n'.join([s['text'] for s in doc_sentences_filtered])      
      s['text'] = context
    return similarities
  
import json

class EvidenceSnippets(EvidenceSelector):
  def __init__(self, urls_directory, include_title=False):
    super(EvidenceSnippets, self).__init__(urls_directory, include_title)    

  def retrieveEvidences(self, claim, claim_sbert, retrieved_documents):    
    json_path = documents_json_path(claim, self.urls_directory)
    f = open(json_path)
    data = json.load(f)
    evidences = []
    for document in retrieved_documents:
      found = False
      for rank, item in enumerate(data['items']):
        if item['link'] == document['found_url']:
          found = True
          if 'snippet' in item:
            snippet = item['snippet']
            evidence = {'url': document['found_url'], 'row': 0, 'text': snippet.strip(),
            'value': 0, 'quote': False, 'doc_has_quotes': 0, 'doc_title': item['title'], 'rank': rank}
            if self.include_title and not evidence['text'].startswith(evidence['doc_title']):
               evidence['text'] = evidence['doc_title'] + "\n" + evidence['text']
            evidences.append(evidence)
          else:
             print("snippet não encontrado. Item:", item)
          break
      if not found:
        print("Erro não encontrado documento:", document['found_url'])
    return evidences[:5]
  
  
def get_evidence_selector(strategy, urls_directory):
  if strategy == EvidenceSelectStrategy.Sentence1:
    return EvidenceSelector(urls_directory)
  elif strategy == EvidenceSelectStrategy.Sentence1NoQuotes:
    return EvidenceSelector(urls_directory, remove_quotes=True)
  elif strategy == EvidenceSelectStrategy.TitleSentence1:
    return EvidenceSelector(urls_directory, include_title=True)
  elif strategy == EvidenceSelectStrategy.Context5:
    return EvidenceSelectorWithContext(urls_directory)
  elif strategy == EvidenceSelectStrategy.Context5NoQuotes:
    return EvidenceSelectorWithContext(urls_directory, remove_quotes=True)
  elif strategy == EvidenceSelectStrategy.TitleContext5:
    return EvidenceSelectorWithContext(urls_directory, include_title=True)
  elif strategy == EvidenceSelectStrategy.TitleContext5NoQuotes:
    return EvidenceSelectorWithContext(urls_directory, include_title=True, remove_quotes=True)
  elif strategy == EvidenceSelectStrategy.OnlyTitle:
    return EvidenceSelector(urls_directory, only_title=True)
  elif strategy == EvidenceSelectStrategy.Snippets:
     return EvidenceSnippets(urls_directory)
  elif strategy == EvidenceSelectStrategy.TitleSnippets:
     return EvidenceSnippets(urls_directory, include_title=True)
  