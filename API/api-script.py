import openai
import pandas as pd

from tqdm import tqdm
tqdm.pandas()


openai.api_key = "Your API Key"
# 读取CSV文件
data = pd.read_csv("D:\\PaperCode\\sepsis\\data\\note_selected.csv")
# 创建一个新列
data["new_text"] = ""
# 循环遍历每一行，取出"text"列中的文本并分析
for index, row in tqdm(data.iterrows(),total=len(data)):
    text = row["text"]
    # 在这里进行你的文本分析0
    max_length = 4096
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # 使用空格截断文本，确保不会截断单词

    rsp = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        messages=[
                  {"role": "system", "content": "Summary Generation Assistant"},
                  {"role": "user", "content": text}
                  ])
    text_pre = rsp.get("choices")[0]['message']['content']
    # 将分析结果添加到新列中
    data.at[index, "new_text"] = text_pre

# 保存结果到CSV文件中
data.to_csv("D:\\PaperCode\\sepsis\\data\\note_processed.csv", index=False)