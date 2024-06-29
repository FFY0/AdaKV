import json
import pandas as pd

# Load the JSON file
file_path = './Log/oldbound_mistral-7B-instruct-v0.2c1024_w32_k7_maxpool_floor0.5_adaptiveTrue_skip0_RWFalse/result.json'
output_path = './Log/oldbound_mistral-7B-instruct-v0.2c1024_w32_k7_maxpool_floor0.5_adaptiveTrue_skip0_RWFalse/output_corrected.xlsx'

with open(file_path, 'r') as file:
    data = json.load(file)

# Create a DataFrame with the specified column order
column_order = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec",
    "triviaqa", "samsum", "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p"
]

# Create a renamed dictionary to match the column order
data_renamed = {
    "narrativeqa": data["narrativeqa"],
    "qasper": data["qasper"],
    "multifieldqa_en": data["multifieldqa_en"],
    "hotpotqa": data["hotpotqa"],
    "2wikimqa": data["2wikimqa"],
    "musique": data["musique"],
    "gov_report": data["gov_report"],
    "qmsum": data["qmsum"],
    "multi_news": data["multi_news"],
    "trec": data["trec"],
    "triviaqa": data["triviaqa"],
    "samsum": data["samsum"],
    "passage_count": data["passage_count"],
    "passage_retrieval_en": data["passage_retrieval_en"],
    "lcc": data["lcc"],
    "repobench-p": data["repobench-p"]
}

# Create the DataFrame
df = pd.DataFrame([data_renamed], columns=column_order)

# Save the DataFrame to an Excel file
df.to_excel(output_path, index=False)

print(f"Excel file saved to {output_path}")
