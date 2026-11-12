import pandas as pd
import argparse

argument_parser = argparse.ArgumentParser(description="Compute the average score from a CSV file.")
argument_parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing the scores.")
args = argument_parser.parse_args()

df = pd.read_csv(args.csv_file)
scores = df["score"]
average_score = scores.mean()
print(f"Average score: {average_score}")