from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from transformers import BertForSequenceClassification
import os
import csv
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
import gc
import torch
from torch.utils.data import DataLoader, SequentialSampler
import sys
import numpy as np

class BertPreprocessor:
  def __init__(self, tokenizer, use_evidence = True):
    self.tokenizer = tokenizer
    self.use_evidence = use_evidence

  def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
      """Truncates a sequence pair in place to the maximum length."""
      # This is a simple heuristic which will always truncate the longer sequence
      # one token at a time. This makes more sense than truncating an equal percent
      # of tokens from each, since if one sequence is very short then each token
      # that's truncated likely contains more information than a longer sequence.
      while True:
          total_length = len(tokens_a) + len(tokens_b)
          if total_length <= max_length:
              break
          if len(tokens_a) > len(tokens_b):
              tokens_a.pop()
          else:
              tokens_b.pop()

  def bert_preprocess_2_Sentences(self, sent1, sent2, max_seq_length=512):
    tokens_a = self.tokenizer.tokenize(sent1)
    tokens_b = self.tokenizer.tokenize(sent2)
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"  
    if sent2:
      truncate_size = max_seq_length - 3
    else:
      truncate_size = max_seq_length - 2

    self._truncate_seq_pair(tokens_a, tokens_b, truncate_size)
    
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    return {'input_ids': input_ids, 'token_type_ids': segment_ids, 'attention_mask': input_mask}

  def tokenizeTexts(self, claims):
    tokenized_texts = []
    
    for i, claim in enumerate(claims):
      if self.use_evidence:
        evidence = claim['evidence_clean']
      else:
        evidence = ''
      tokenized_texts.append(self.bert_preprocess_2_Sentences(claim['claim_clean'], evidence))
    return tokenized_texts

class LanguageModelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
      return self.encodings[idx]['input_ids']

    def __len__(self):
      return len(self.encodings)

class ClassifierDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels, ids = []):
    self.encodings = encodings
    self.labels = labels
    self.ids = ids

  def __getitem__(self, idx):
    item = {}
    item['input_ids'] = torch.tensor(self.encodings[idx]['input_ids'])
    item['token_type_ids'] = torch.tensor(self.encodings[idx]['token_type_ids'])
    item['attention_mask'] = torch.tensor(self.encodings[idx]['attention_mask'])
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

class BertModel:
  def __init__(self, model_base, tokenizer_name, labels, use_evidence = True, resume_from_checkpoint = False, device = 'cuda', task = 'classification'):
    print('model_base:', model_base,
          '\ntokenizer_name:',tokenizer_name,
          '\nlabels:', labels, 
          '\nuse_evidence:', use_evidence,
          '\nresume_from_checkpoint:', resume_from_checkpoint,
          '\ndevice:', device, 
          '\ntask:', task)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    self.labels = labels
    self.device = device
    self.resume_from_checkpoint = resume_from_checkpoint
    self.task = task
    if task == 'classification':
      self.model = BertForSequenceClassification.from_pretrained(model_base, num_labels = len(labels))
    else:
      self.model  = BertForMaskedLM.from_pretrained(model_base)
    self.model.to(device)
    self.preprocessor = BertPreprocessor(tokenizer, use_evidence)

  def __loadSamples(self, filename):
    claims = []
    with open(filename,'r', encoding='utf-8') as f:
      read_claims = csv.DictReader(f, delimiter="\t", skipinitialspace=True)
      i = 0
      for claim in read_claims: 
        claim['rowclaim'] = i
        if int(claim['id']) > 0:
          claim['idclaim'] = int(claim['id'])
        else:
          claim['idclaim']=int(abs(int(claim['id'])/10))
        claims.append(claim)
        i += 1
    return claims

  def get_tokens(self, claimtxt, evidencetxt):
    samples = [{'claim_clean': claimtxt, 'evidence_clean': evidencetxt}]
    X = self.preprocessor.tokenizeTexts(samples)[0]
    b_input_ids = torch.tensor([X['input_ids']]).to(self.device)
    b_token_type_ids = torch.tensor([X['token_type_ids']]).to(self.device)
    b_input_mask = torch.tensor([X['attention_mask']]).to(self.device)
    size = 0
    for mask in b_input_mask[0]:
      if mask == 1:
        size += 1
    return self.preprocessor.tokenizer.convert_ids_to_tokens(b_input_ids[0][:size])

  def predict(self, claimtxt, evidencetxt):
    samples = [{'claim_clean': claimtxt, 'evidence_clean': evidencetxt}]
    X = self.preprocessor.tokenizeTexts(samples)[0]
    b_input_ids = torch.tensor([X['input_ids']]).to(self.device)
    b_token_type_ids = torch.tensor([X['token_type_ids']]).to(self.device)
    b_input_mask = torch.tensor([X['attention_mask']]).to(self.device)

    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = self.model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      pred_ids = np.argmax(logits, axis=1).flatten()      
      return pred_ids[0]

  def createDataset(self, path):
    samples = self.__loadSamples(path)
    X = self.preprocessor.tokenizeTexts(samples)
    if self.task == 'classification':
      y = [self.labels.index(sample['class']) for sample in samples]
      ids = [sample['id'] for sample in samples]
      dataset = ClassifierDataset(X, y, ids)
    else:
      dataset = LanguageModelDataset(X)
    return dataset
  
  def fineTune(self, dataset, model_name):
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
      tokenizer=self.preprocessor.tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
      output_dir='./results',          # output directory
      num_train_epochs=3,              # total # of training epochs
      per_device_train_batch_size=2,  # batch size per device during training
      #per_device_train_batch_size=32,  # batch size per device during training (bert fever)
      per_device_eval_batch_size=64,   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs            #'--learning_rate', '2e-5'
      save_total_limit = 3, 
      resume_from_checkpoint=self.resume_from_checkpoint
    )
    trainer = Trainer(
      model=self.model,                         # the instantiated  Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=dataset,         # training dataset
      data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
    trainer.save_model(model_name)


  def train(self, dataset_train, dataset_valid, model_name):
    training_args = TrainingArguments(
      output_dir='./results',          # output directory
      num_train_epochs=3,              # total # of training epochs
      per_device_train_batch_size=2,  # batch size per device during training
      #per_device_train_batch_size=32,  # batch size per device during training (bert fever)
      per_device_eval_batch_size=64,   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs            #'--learning_rate', '2e-5'
      save_total_limit = 3, 
      resume_from_checkpoint=self.resume_from_checkpoint
    )

    trainer = Trainer(
      model=self.model,                         # the instantiated Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=dataset_train,         # training dataset
      eval_dataset=dataset_valid            # evaluation dataset
    )
    trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
    trainer.save_model(model_name) #Duração 48m e 31s

  def eval(self, dataset_test, log_file):
    self.model.eval()
    
    batch_size = 32

    test_dataloader = DataLoader(
      dataset_test, # The validation samples.
      sampler = SequentialSampler(dataset_test), # Pull out batches sequentially.
      batch_size = batch_size # Evaluate with this batch size.
    )

    predictions , true_labels = [], []

    i = 0
    # Predict 
    for batch in test_dataloader:
      print(i)
      i+=1
      
      # Unpack the inputs from our dataloader
      #print(batch)
      b_input_ids = batch['input_ids'].to(self.device)
      b_token_type_ids = batch['token_type_ids'].to(self.device)
      b_input_mask = batch['attention_mask'].to(self.device)
      b_labels = batch['labels'].to(self.device)
      #, b_token_type_ids, b_input_mask, b_labels = batch
      #print(b_input_ids)

      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction
      with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = self.model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      pred_ids = np.argmax(logits, axis=1).flatten()
      label_ids = b_labels.to('cpu').numpy()    
      # Store predictions and true labels
      predictions.extend(pred_ids)
      true_labels.extend(label_ids)

    with open(log_file, 'w', encoding="utf-8", newline='') as file:
      f = csv.writer(file, delimiter=';')
      f.writerow(['id', 'true_label', 'prediction'])
      for i in range(len(true_labels)):                
        f.writerow([dataset_test.ids[i], true_labels[i], predictions[i]])
      file.close()

  def delete(self):  
    del self.model
    self.model = None
    gc.collect()

    with torch.no_grad():
      torch.cuda.empty_cache()

def trainBertClassifier(path_train, path_valid, path_test, model_name, use_evidence = True, resume_from_checkpoint = False, base_model='neuralmind/bert-base-portuguese-cased'):
  print("Training. train=", path_train, " valid=", path_valid, " test=", path_test, "model=", model_name)
  model = BertModel(base_model,'neuralmind/bert-base-portuguese-cased', ['VERDADEIRO', 'FALSO', 'INSUFICIENTE'], use_evidence, resume_from_checkpoint)
  print("Creating training dataset:")
  dataset_train = model.createDataset(path_train)  
  print("Creating validation dataset:")
  dataset_valid = model.createDataset(path_valid)
  print("Creating test dataset:")
  dataset_test = model.createDataset(path_test)
  print("Treining:")
  model.train(dataset_train, dataset_valid, model_name)
  print("Evaluation:")
  model.eval(dataset_valid, 'log_' + model_name.replace("/","_").replace("\\","_") + '_valid.txt')
  model.eval(dataset_test, 'log_' + model_name.replace("/","_").replace("\\","_") + '_test.txt')
  return model

def evalBertClassifier(model_name, tokenizer_name, path_test, out_file, use_evidence = True):
  model = BertModel(model_name, tokenizer_name, ['VERDADEIRO', 'FALSO', 'INSUFICIENTE'], use_evidence)
  dataset_test = model.createDataset(path_test)
  model.eval(dataset_test, out_file)
  return model