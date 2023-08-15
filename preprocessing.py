import re
import nltk
nltk.download('punkt')

def remove_emoji(string):
  emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0000231A-\U0000231B"
    u"\U0001F90D-\U0001FA9F"
    u"\U0001F7E0-\U0001F7EB"
    u"\U0000200D"
    "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'. ', string)
  

def normalize(text):
  text = text.replace('&#39;', "")
  text = re.sub(r'(\d+)\.(\d+)',r'\1\2', text) #removing thousand separator
  text = remove_emoji(text)
  text = text.replace('\n', ' . ')
  text = re.sub(r';\s*([A-Z])',r'. \1', text) # ; followed by a capital letter, which probably indicates the end of a sentence
  text = text.replace('•','. ') #separating topics into sentences
  text = re.sub(r';\s*\.',r'.', text) # ; .
  text = text.replace('; –','. ') #separating topics into sentences  
  text = text.replace('; -','. ') #separating topics into sentences  
  text = re.sub(r'^\s*–','', text) #removing topics
  text = re.sub(r'^\s*-','', text) #removing topics
  text = text.replace('- ',' ') #removing topics
  text = re.sub(r'\d+ –','', text) #removing numbering
  text = text.replace('– ',' ') #removing topics
  text = re.sub(r'#','', text) # removing hashtags
  text = re.sub(r'\s+',' ', text) #removing duplicate spaces
  text = text.replace('\\n', ' . ') #removing line break
  text = text.replace("\\t", ' ') #removing tab
  text = text.replace('\u202c', ' ') #unknown characters
  text = text.replace('\u202a', ' ') #unknown characters
  text = text.replace("”", '"')
  text = text.replace("“", '"')
  text = text.replace('./ ', '. ')
  text = re.sub(r'///+',' . ', text)
  text = text.replace('(…)','')
  text = text.replace('[','')
  text = text.replace(']','')
  text = text.replace('None ','')
  text = text.replace('\x00', '')
  text = text.strip()

  return text
  
def tokenize(text, remove_obvious = False):
  normaltext = normalize(text)
  a_list = nltk.tokenize.sent_tokenize(normaltext, 'portuguese')
  list_clean = []
  is_obvious = False
  
  obvious_evidences = [' analisada pela Lupa é falsa', ' analisado pela Lupa é falsa',
    ' verificada pela Lupa é falsa', ' checada pela Lupa é falsa', ' analisado pela Lupa é falso',
    ' verificado pela Lupa é falso', ' checado pela Lupa é falso']
  for text in a_list:
    
    is_obvious = False
    if remove_obvious:
      for ev in obvious_evidences:
        if ev in text:
          is_obvious = True
          break
    text = re.sub(r'https:\S+','', text) #removing url. It was done after tokenize because sometimes it has punctuation after the url
    text = re.sub(r'http:\S+','', text) #removing url
    text = re.sub(r'saiba mais.*','', text, flags=re.IGNORECASE)
    text = re.sub(r'Leia íntegra.*','', text)
    text = text.replace('(...)','') 
    if text == 'None':
      text = ''
    text = text.replace(' .', '')
    text = text.strip()    
    if text != '' and len(text) > 1 and not is_obvious:
      list_clean.append(text)
  return list_clean

def tokenize_and_join(text, remove_obvious = False):
  tokens = tokenize(text, remove_obvious)
  return '\n'.join(tokens)