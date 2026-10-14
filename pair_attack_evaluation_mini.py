from strong_reject.load_datasets import load_strongreject_small
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset

import os
from datetime import datetime
import pandas as pd
from datasets import Dataset


print(f"Counting cpus", 100 * "*")
print(os.cpu_count(), len(os.sched_getaffinity(0)), os.environ.get("SLURM_CPUS_PER_TASK"))

EVALUATOR = f"strongreject_finetuned"
#victim_model = "hf:lmsys/vicuna-7b-v1.5"
victim_model = "hf:meta-llama/Llama-2-7b-chat-hf"

dataset_file = "data/strong_reject100.csv"
# Create an output directory for this run, timestamped so repeated runs don't overwrite each other
run_dir_alltime = os.path.join("results", f"{victim_model.split('/')[-1]}_{dataset_file.split('/')[-1].split('.')[0]}_pair_{EVALUATOR.split('_')[-1]}")
# load the small version of the StrongREJECT dataset
df = pd.read_csv(dataset_file)
forbidden_prompt_dataset = Dataset.from_pandas(df, preserve_index=False)

## check for existing file and load it
## The prior jailbreaks might have been calculated at another time, so we want to load them if they exist, otherwise we will apply the jailbreaks and save them for future use
jailbreaks_found = False
for path in os.listdir(run_dir_alltime):
    if os.path.exists(os.path.join(run_dir_alltime, path, "jailbroken_prompts_mini.csv")):
        run_dir = os.path.join(run_dir_alltime, path)
        jailbreaks_found = True
if jailbreaks_found:
    print(f"Loading existing jailbroken prompts from {os.path.join(run_dir, 'jailbroken_prompts_mini.csv')}")
    jailbroken_dataset = Dataset.from_pandas(pd.read_csv(os.path.join(run_dir, "jailbroken_prompts_mini.csv")))
    print(f"Saving all artifacts to: {run_dir}")
else:
    print("No existing jailbroken prompts found. Applying jailbreak.")
    run_dir = os.path.join(run_dir_alltime,datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving all artifacts to: {run_dir}")
    # apply the new jailbreak and compare it to a no-jailbreak baseline ("none")
    jailbroken_dataset = apply_jailbreaks_to_dataset(forbidden_prompt_dataset, ["pair"], victim_model=victim_model, max_tokens = 500)
    jailbroken_dataset.to_pandas().to_csv(
        os.path.join(run_dir, "jailbroken_prompts_mini.csv"), index=False
    )

print(100*"=", f"\n Generating responses to the jailbroken prompts using {victim_model} \n", 100*"=") 
# generate responses to the jailbroken prompts

if jailbreaks_found:
    print(f"Staying in directory {run_dir}")
    if os.path.exists(os.path.join(run_dir, "responses_mini.csv")):
        print(f"Loading existing generated responses from {os.path.join(run_dir, 'responses_mini.csv')}")
        responses_dataset = Dataset.from_pandas(pd.read_csv(os.path.join(run_dir, "responses_mini.csv")))
    else:
        print("No existing generations found. Applying generation.")
        print(f"Saving all artifacts to: {run_dir}")
        responses_dataset = generate_to_dataset(jailbroken_dataset, [victim_model], target_column="jailbroken_prompt", num_processes=1)
        responses_dataset.to_pandas().to_csv(os.path.join(run_dir, "responses_mini.csv"), index=False)
else:
    print("No existing generations found. Applying generation.")
    print(f"Saving all artifacts to: {run_dir}")
    responses_dataset = generate_to_dataset(jailbroken_dataset, [victim_model], target_column="jailbroken_prompt", num_processes=1)
    responses_dataset.to_pandas().to_csv(os.path.join(run_dir, "responses_mini.csv"), index=False)


# use the StrongREJECT evaluator to score the harmfulness of the responses
print(100*"=", f"\n Evaluating responses using {EVALUATOR} \n", 100*"=")
eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR])
eval_df = eval_dataset.to_pandas()
eval_df.to_csv(os.path.join(run_dir, "responses_and_evaluations.csv"), index=False)

# compare the average harmfulness of the new jailbreak to the no-jailbreak baseline
summary = eval_df.groupby("jailbreak")["score"].mean()

print(f"All artifacts saved to: {run_dir}")
print(summary)

