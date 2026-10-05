from strong_reject.load_datasets import load_strongreject_small
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset

import os
from datetime import datetime
import pandas as pd
from datasets import Dataset

EVALUATOR = f"strongreject_finetuned"
victim_model = "hf:lmsys/vicuna-7b-v1.5"
dataset_file = "data/strong_reject100.csv"
# Create an output directory for this run, timestamped so repeated runs don't overwrite each other
run_dir = os.path.join("results", f"{victim_model.split("/")[-1]}_{dataset_file.split("/")[-1].split(".")[0]}_{EVALUATOR.split('_')[-1]}", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

#TODO have this load from a permanent set of jailbreaks
# load the small version of the StrongREJECT dataset
df = pd.read_csv(dataset_file)
forbidden_prompt_dataset = Dataset.from_pandas(df, preserve_index=False)

# apply the new jailbreak and compare it to a no-jailbreak baseline ("none")
jailbroken_dataset = apply_jailbreaks_to_dataset(forbidden_prompt_dataset, ["pair"], victim_model=victim_model, max_tokens = 500)
jailbroken_dataset.to_pandas().to_csv(
    os.path.join(run_dir, "jailbroken_prompts.csv"), index=False
)

# generate responses to the jailbroken prompts
responses_dataset = generate_to_dataset(jailbroken_dataset, [victim_model], target_column="jailbroken_prompt")
# We won't save
#pd_responses_dataset.to_csv(os.path.join(run_dir, "responses.csv"), index=False)

# use the StrongREJECT evaluator to score the harmfulness of the responses
eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR])
eval_df = eval_dataset.to_pandas()
eval_df.to_csv(os.path.join(run_dir, "responses_and_evaluations.csv"), index=False)

# compare the average harmfulness of the new jailbreak to the no-jailbreak baseline
summary = eval_df.groupby("jailbreak")["score"].mean()

print(f"All artifacts saved to: {run_dir}")
print(summary)

