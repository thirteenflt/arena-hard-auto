import pandas as pd
import numpy as np
import plotly.express as px

import tiktoken
import datetime
import argparse
import os
import math
import re

from glob import glob
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from utils import load_model_answers
import jsonlines
from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    args = parser.parse_args()

    data_dir = os.path.join(f"data/{args.bench_name}/model_judgment", args.judge_name)
    data_files = glob(os.path.join(data_dir, "*.jsonl"))

    for data_file in data_files:
        model_name = data_file.split('/')[-1].split('.')[0]
        print(f"================== {model_name} ====================")

        # Parse the JSONL data
        json_data = []
        with jsonlines.open(data_file) as reader:
            for obj in reader:
                json_data.append(obj)

        # Initialize a dictionary to store pass and fail rates
        results = {}

        # Process each entry in the JSONL data
        for entry in json_data:
            # Extract language and category
            question_id = entry['question_id']
            language = question_id.split('[')[1].split(']')[0]
            category = entry['category']
            
            # Extract the score
            score = entry['games'][0]['score']
            
            # Initialize the language and category in the results dictionary if not already present
            if language not in results:
                results[language] = {}
            if category not in results[language]:
                results[language][category] = {'Pass': 0, 'Fail': 0, 'Missing': 0}
            
            # Update the pass or fail count
            if score == 'Pass':
                results[language][category]['Pass'] += 1
            elif score == 'Fail':
                results[language][category]['Fail'] += 1
            else:
                results[language][category]['Missing'] += 1

        # Print the results beautifully
        table_data = []
        for language, categories in results.items():
            for category, counts in categories.items():
                total = counts['Pass'] + counts['Fail'] + counts['Missing']
                pass_rate = counts['Pass'] / total * 100 if total > 0 else 0
                fail_rate = counts['Fail'] / total * 100 if total > 0 else 0
                missing_rate = counts['Missing'] / total * 100 if total > 0 else 0
                table_data.append([language, category, f"{pass_rate:.2f}%", f"{fail_rate:.2f}%", f"{missing_rate:.2f}%"])
        print(tabulate(table_data, headers=["Language", "Category", "Pass Rate", "Fail Rate", "Missing Rate"]))