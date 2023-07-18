import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
tqdm.pandas()
import time
import random
random.seed(2023)

# 加载ClinicalBERT模型和分词器
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 读取原始数据
data = pd.read_csv('Your note csv')
subject_ids = data['subject_id']
chartdate = data['chartdate']
charttime = data['charttime']
category = data['category']


# 定义结果列表
results = []

# 逐行处理每个文本
for index, row in tqdm(data.iterrows(),total=len(data)):
    text = row['text']
    subject_id = row['subject_id']

    # 将文本转化为token并进行截断和填充
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # 将token输入ClinicalBERT模型
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # 将结果存入列表
    results.append(embeddings)

# 将结果保存到CSV文件中
df = pd.DataFrame(results)
df['subject_id'] = subject_ids
df['chartdate']  = chartdate
df['charttime'] = charttime
df['category'] = category



df.to_csv("Generated Vector CSV", index=False)
