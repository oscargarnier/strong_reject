from strong_reject.load_datasets import load_strongreject_small
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset

import os
from datetime import datetime
import pandas as pd
from datasets import Dataset
from ressources.models import hugging_face_model_adresses
EVALUATOR = f"strongreject_finetuned"
victim_model = hugging_face_model_adresses["llama2-7"]
dataset_file = "data/strong_reject3.csv"
attack = "gcg_transfer_harmbench"

# load the StrongREJECT dataset
df = pd.read_csv(dataset_file)
forbidden_prompt_dataset = Dataset.from_pandas(df, preserve_index=False)
run_dir_notime = os.path.join("results", f"{victim_model.split('/')[-1]}_{dataset_file.split('/')[-1].split('.')[0]}_{attack}_{EVALUATOR.split('_')[-1]}")
os.makedirs(run_dir_notime, exist_ok=True)
## The prior jailbreaks might have been calculated at another time, so we want to load them if they exist, otherwise we will apply the jailbreaks and save them for future use
jailbreaks_found = False
for path in os.listdir(run_dir_notime):
    # get the latest directory with a jailbroken_prompts.csv file
    if os.path.exists(os.path.join(run_dir_notime, path, "jailbroken_prompts.csv")):
        run_dir = os.path.join(run_dir_notime, path)
        jailbreaks_found = True
if jailbreaks_found:
    print(f"Loading existing jailbroken prompts from {os.path.join(run_dir, 'jailbroken_prompts.csv')}")
    jailbroken_dataset = Dataset.from_pandas(pd.read_csv(os.path.join(run_dir, "jailbroken_prompts.csv")))
    print(f"Saving all artifacts to: {run_dir}")
else:
    print("No existing jailbroken prompts found. Applying jailbreak.")
    run_dir = os.path.join(run_dir_notime,datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving all artifacts to: {run_dir}")
    # apply the new jailbreak
    jailbroken_dataset = apply_jailbreaks_to_dataset(forbidden_prompt_dataset, [attack], victim_model=victim_model, max_tokens = 500)
    jailbroken_dataset.to_pandas().to_csv(
        os.path.join(run_dir, "jailbroken_prompts.csv"), index=False
    )

# generate responses to the jailbroken prompts
print(100*"=", f"\n Generating responses to the jailbroken prompts using {victim_model} \n", 100*"=") 

if os.path.exists(os.path.join(run_dir, "responses.csv")):
    print(f"Loading existing generated responses from {os.path.join(run_dir, 'responses.csv')}")
    responses_dataset = Dataset.from_pandas(pd.read_csv(os.path.join(run_dir, "responses.csv")))
else:
    print("No existing generations found. Applying generation.")
    responses_dataset = generate_to_dataset(jailbroken_dataset, [victim_model], target_column="jailbroken_prompt", num_processes=1)
    responses_dataset.to_pandas().to_csv(os.path.join(run_dir, "responses.csv"), index=False)
    print(f"Saving all artifacts to: {run_dir}")

# use the StrongREJECT evaluator to score the harmfulness of the responses
print(100*"=", f"\n Evaluating responses using {EVALUATOR} \n", 100*"=")

if os.path.exists(os.path.join(run_dir, "responses_and_evaluations.csv")):
    print(f"Loading existing evaluations from {os.path.join(run_dir, 'responses_and_evaluations.csv')}")
    eval_df = pd.read_csv(os.path.join(run_dir, "responses_and_evaluations.csv"))
else:
    print("No existing evaluations found. Applying evaluation.")
    print(f"Saving all artifacts to: {run_dir}")
    eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR])
    eval_df = eval_dataset.to_pandas()
    eval_df.to_csv(os.path.join(run_dir, "responses_and_evaluations.csv"), index=False)

# compare the average harmfulness of the new jailbreak to the no-jailbreak baseline
summary = eval_df.groupby("jailbreak")["score"].mean()

print(f"All artifacts saved to: {run_dir}")
print(summary)


