from enum import Enum

class ClassifierStrategy(Enum):
  Bert3 = 1
  FindWords = 2

class EvidenceSelectStrategy(Enum):
  Sentence1 = 1
  Sentence1NoQuotes = 2
  TitleSentence1 =3
  Context5 = 4
  Context5NoQuotes = 5
  TitleContext5 = 6
  TitleContext5NoQuotes = 7
  OnlyTitle = 8  
  Snippets = 9
  TitleSnippets = 10