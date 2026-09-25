from strong_reject.load_datasets import load_strongreject_small
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset

import os
from datetime import datetime

EVALUATOR = "strongreject_finetuned"
RESPONSES_FILE = "responses/llama2_jbb.csv" 
experiment_name = "llama_jbb"

run_dir = os.path.join("results", experiment_name,datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

import pandas as pd
from datasets import Dataset


## loads precomputed responses
df = pd.read_csv(RESPONSES_FILE)
responses_dataset = Dataset.from_pandas(df, preserve_index=False)

## evaluates these responses
eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR])
eval_df = eval_dataset.to_pandas()
eval_df.to_csv(os.path.join(run_dir, "evaluation_results.csv"), index=False)

# compare the average harmfulness of the new jailbreak to the no-jailbreak baseline
summary = eval_df.groupby("jailbreak")["score"].mean()
summary.to_csv(os.path.join(run_dir, "summary_by_jailbreak.csv"))

print(f"All artifacts saved to: {run_dir}")
print(summary)

