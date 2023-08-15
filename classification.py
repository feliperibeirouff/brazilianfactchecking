import csv
from strategies import ClassifierStrategy

sites_fact_checking = [
    'https://piaui.folha.uol.com.br/lupa/',
    'https://g1.globo.com/fato-ou-fake/',
    'https://www.aosfatos.org/',
    'https://www.boatos.org/',
    'https://www.e-farsas.com/',
    'https://politica.estadao.com.br/blogs/estadao-verifica',
    'https://checamos.afp.com/',
    'https://noticias.uol.com.br/confere/',
    'https://projetocomprova.com.br/',
    'https://apublica.org/checagem'    
]

SUPORTA = 'SUPORTA'
REFUTA = 'REFUTA'
INSUFICIENTE = 'INSUFICIENTE'

model = None

def isFactCheckingUrl(url):
    for site in sites_fact_checking:
        if site in url:
            return True
    return False

def bertClassify(claimtxt, evidencetxt):
  global model
  if model == None:
    from bert_training import BertModel
    model = BertModel('models/base3_finetuned_3classes', 'neuralmind/bert-base-portuguese-cased', ['VERDADEIRO', 'FALSO', 'INSUFICIENTE'])
  predict = model.predict(claimtxt, evidencetxt)  
  return [SUPORTA, REFUTA, INSUFICIENTE][predict]

def findWordsClassify(claimtxt, evidencetxt):
  lowerEv = evidencetxt.lower()
  refutesWords = ['fake',
                  'falso', 'falsa', 'falsos', 'falsas',
                  'mentira', 'mentiras',
                  'calúnia', 'calúnias',
                  'inverídico', 'inverídica', 'inverídicos', 'inverídicas',
                  'enganoso', 'enganosa', 'enganosos', 'enganosas',
                  'farsa', 'farsas',
                  'ilusório', 'ilusória', 'ilusórios', 'ilusórias',
                  'ilegítimo', 'ilegítima', 'ilegítimos', 'ilegítimas',
                  'boato', 'boatos',
                  'rumor', 'rumores'
                ]
  for r in refutesWords:
    if r in lowerEv:
      return REFUTA
  return SUPORTA

def findEvidences(claim, allEvidences):
  id = claim['id']
  return [ev for ev in allEvidences if ev['claim_id'] == id]

def checkClaim(claim, allEvidences, classifier_method):
  retrieved_evidences = findEvidences(claim, allEvidences)
  claimtxt = claim['claim_clean']
  list_dic = []
  for evidence in retrieved_evidences:
    evidencetxt = evidence['evidence_clean']
    title = evidence['doc_title']
    url = evidence['found_url']
    if isFactCheckingUrl(url):
      evidence['verified_by_factchecking_site'] = True
    else:
      evidence['verified_by_factchecking_site'] = False

    if classifier_method == ClassifierStrategy.Bert3:
      evidence['label'] = bertClassify(claimtxt, evidencetxt)
      evidence['tokens'] = model.get_tokens(claimtxt, evidencetxt)
    elif classifier_method == ClassifierStrategy.FindWords:
      evidence['label'] = findWordsClassify(claimtxt, evidencetxt)
      evidence['tokens'] = ''
    else:
      print("ERROR. Method not found")

    dic = {'similarity': evidence['similarity'], 'predicted_label': evidence['label'], 
         'evidence': evidencetxt, 'doc_rank': evidence['doc_rank'], 'url': url, 'doc_title': evidence['doc_title'],
         'claim_id': claim['id'], 'claimtxt': claimtxt, 'claim_class': claim.get('class','N/A'), 'quote': evidence['quote'],
         'verified_by_factchecking_site': evidence['verified_by_factchecking_site'], 'qtd_tokens': len(evidence['tokens']), 'tokens': ' '.join(evidence['tokens'])}
    list_dic.append(dic)
  
  label = INSUFICIENTE
  for evidence in retrieved_evidences:
    if evidence['label'] == REFUTA:
      label = REFUTA
      break
    elif evidence['label'] == SUPORTA:
      label = SUPORTA
  claim['predicted_label'] = label  
  return list_dic

def classify_claims(claims, evidences, evidences_classified_path, claims_classified_path, classifier_method):
  with open(evidences_classified_path,'w', encoding='utf-8', newline="") as f:
    fieldnames = ['claim_id', 'claim_class', 'similarity', 'predicted_label', 'claimtxt', 'evidence', 'doc_title', 'doc_rank', 'url', 'quote', 'verified_by_factchecking_site', 'qtd_tokens', 'tokens']
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()   
    for row, claim in enumerate(claims):
      if row % 150 == 0:
        print(row)
      evidences_checked = checkClaim(claim, evidences, classifier_method)
      for e in evidences_checked:
        writer.writerow(e)

  with open(claims_classified_path,'w', encoding='utf-8', newline="") as f:
    fieldnames = ['claim_id', 'claim_class', 'predicted_label']
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader() 
    for claim in claims:
      writer.writerow({'claim_id': claim['id'], 'claim_class': claim.get('class','N/A'), 'predicted_label': claim['predicted_label']})

