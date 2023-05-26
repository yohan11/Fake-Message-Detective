import pandas as pd

file_name='/Users/junyeong/Desktop/DKU/2023/1st/Capstone/Fake-Message-Detective/fake_message_detective/main/training_data1.xlsx'

df = pd.read_excel(file_name)

banmal = []
jondae = []

for c in df.columns:
     if '반말' in c :
          for sentence in df[c].astype(str):
               banmal.append(sentence)
     
for c in df.columns:
     if '존댓말' in c :
          for sentence in df[c].astype(str):
               jondae.append(sentence)

dataset_sentences = banmal + jondae

dataset_labels = [0] * len(banmal) + [1] * len(jondae)

