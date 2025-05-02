import json
from collections import Counter
import numpy as np
import glob
from helper_prepare_knowledge import fetch_fomc_statement
from sentence_transformers import SentenceTransformer, util

meeting_code = "22_6"

code2date = {
    "18_1": "January 2018",
    "18_3": "March 2018",
    "18_5": "May 2018",
    "18_6": "June 2018",
    "18_8": "August 2018",
    "18_9": "September 2018",
    "18_11": "November 2018",
    "18_12": "December 2018",
    "22_6": "June 2022",
    "23_2": "Feb 2023",
    "23_3": "March 2023",
    "23_5": "May 2023",
    "23_6": "June 2023",
    "23_7": "July 2023",
    "23_9": "September 2023",
    "23_11": "November 2023",
    "23_12": "December 2023",
    "24_1": "January 2024",
    "24_3": "March 2024",
    "24_5": "May 2024",
    "24_6": "June 2024",
    "24_7": "July 2024",
    "24_9": "September 2024",
    "24_11": "November 2024",
    "24_12": "December 2024",
    "25_3": "March 2025",
}

code2fomcdate = {
    "18_1": 20180131,
    "18_3": 20180321,
    "18_5": 20180502,
    "18_6": 20180613,
    "18_8": 20180801,
    "18_9": 20180926,
    "18_11": 20181108,
    "18_12": 20181219,
    "22_6": 20220615,
    "23_2": 20230201,
    "23_3": 20230322,
    "23_5": 20230503,
    "23_6": 20230614,
    "23_7": 20230726,
    "23_9": 20230920,
    "23_11": 20231101,
    "23_12": 20231213,
    "24_1": 20240131,
    "24_3": 20240320,
    "24_5": 20240501,
    "24_6": 20240612,
    "24_7": 20240731,
    "24_9": 20240918,
    "24_11": 20241107,
    "24_12": 20241218,
    "25_3": 20250319,
}

code2vote = {
    "18_1": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "18_3": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "18_5": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "18_6": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "18_8": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "18_9": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "18_11": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "18_12": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "22_6": ["0.75%", "+0.75%", "0.75", "+0.75"],
    "23_2": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "23_3": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "23_5": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "23_6": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "23_7": ["0.25%", "+0.25%", "0.25", "+0.25"],
    "23_9": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "23_11": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "23_12": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "24_1": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "24_3": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "24_5": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "24_6": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "24_7": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
    "24_9": ["-0.50%", "-0.50"],
    "24_11": ["-0.25%", "-0.25"],
    "24_12": ["-0.25%", "-0.25"],
    "25_3": ["0.00%", "-0.00%", "+0.00%", "0.00", "-0.00", "+0.00"],
}

code2pred = {
    "23_2": [
        "3.00%-3.25%",
        "3.00%",
        "3.25%",
        "3.00-3.25%",
        "3.00",
        "3.25",
        "3.00-3.25",
        "2.50%-2.75%",
        "2.50%",
        "2.75%",
        "2.50-2.75%",
        "2.50",
        "2.75",
        "2.50-2.75",
    ],
    "23_5": [
        "2.75%-3.00%",
        "2.75%",
        "3.25%",
        "2.75-3.00%",
        "2.75",
        "3.25",
        "2.75-3.00",
        "3.00%-3.25%",
        "3.00%",
        "3.00-3.25%",
        "3.00",
        "3.00-3.25",
        "2.75%-3.25%",
        "2.75-3.25%",
        "2.75-3.25",
    ],
    "23_6": [
        "2.75%-3.00%",
        "2.75%",
        "3.25%",
        "2.75-3.00%",
        "2.75",
        "3.25",
        "2.75-3.00",
        "3.00%-3.25%",
        "3.00%",
        "3.00-3.25%",
        "3.00",
        "3.00-3.25",
        "2.75%-3.25%",
        "2.75-3.25%",
        "2.75-3.25",
    ],
    "23_7": [
        "3.25%-3.50%",
        "3.25%",
        "3.50%",
        "3.25-3.50%",
        "3.25",
        "3.50",
        "3.25-3.50",
        "3.00%-3.25%",
        "3.00%",
        "3.00-3.25%",
        "3.00",
        "3.00-3.25",
        "3.00%-3.50%",
        "3.00-3.50%",
        "3.00-3.50",
    ],
    "24_3": [
        "3.75%-4.00%",
        "3.75%",
        "4.00%",
        "3.75-4.00%",
        "3.75",
        "4.00",
        "3.75-4.00",
    ],
    "24_9": [
        "3.25%-3.50%",
        "3.25%",
        "3.50%",
        "3.25-3.50%",
        "3.25",
        "3.50",
        "3.25-3.50",
    ],
    "24_11": [
        "3.25%-3.50%",
        "3.25%",
        "3.50%",
        "3.25-3.50%",
        "3.25",
        "3.50",
        "3.25-3.50",
    ],
    "24_12": [
        "3.75%-4.00%",
        "3.75%",
        "4.00%",
        "3.75-4.00%",
        "3.75",
        "4.00",
        "3.75-4.00",
    ],
    "25_3": [
        "3.75%-4.00%",
        "3.75%",
        "4.00%",
        "3.75-4.00%",
        "3.75",
        "4.00",
        "3.75-4.00",
    ],
}
meeting_date = code2date[meeting_code]
meeting_date_fomc = code2fomcdate[meeting_code]
correct_vote = code2vote[meeting_code]
# correct_pred = code2pred[meeting_code]

# Step 1: Get a list of all JSON files
json_files = glob.glob("rate_summary*.json")

# Step 2: Read all JSON files into a list
rate_summaries = []


for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        rate_summaries.append(data)  # Append each JSON object to the list


# Dictionary to store votes by member
votes_by_member = {}
# predictions_by_member = {}
all_votes = []
# all_preds = []
# all_dates = []
# all_metrics = []
all_reasonings = []

# Loop through each JSON object
for summary in rate_summaries:
    if "fomc_public_statement" in summary:
        all_reasonings.append(summary["fomc_public_statement"])

    for vote in summary["rate_votes"]:
        member = vote["member"]
        vote_value = vote["vote"]

        # Initialize list if member not in dictionary
        if member not in votes_by_member:
            votes_by_member[member] = []

        # Append vote to the corresponding member
        votes_by_member[member].append(vote_value)
        all_votes.append(vote_value)
    """for prediction in summary["rate_predictions"]:
        member = prediction["member"]
        prediction_value = prediction["prediction"]

        # Initialize list if member not in dictionary
        if member not in predictions_by_member:
            predictions_by_member[member] = []

        # Append vote to the corresponding member
        predictions_by_member[member].append(prediction_value)
        all_preds.append(prediction_value)
        """
    # all_dates.append(summary["exact_historical_dates_referenced"])
    # all_metrics.append(summary["exact_metrics_mentioned"])

vote_agreement_rates = []
# prediction_agreement_rates = []

for member in votes_by_member:
    vote_counts = Counter(votes_by_member[member])
    # Get the most common value and its count
    most_common_value, count = vote_counts.most_common(1)[0]
    vote_agreement_rates.append(100 * count / len(votes_by_member[member]))
    # Do the same for predictions
    # prediction_counts = Counter(predictions_by_member[member])
    # Get the most common value and its count
    # most_common_value, count = prediction_counts.most_common(1)[0]
    # prediction_agreement_rates.append(100 * count / len(predictions_by_member[member]))


# print(vote_agreement_rates)
avg_vote_agreement = np.mean(vote_agreement_rates)
# print("Imported fetch_fomc_statement:", fetch_fomc_statement)
print(f"Meeting Date: {meeting_date}")
print(f"average agreement rate for votes: {avg_vote_agreement}%")
# print(prediction_agreement_rates)
# avg_prediction_agreement = np.mean(prediction_agreement_rates)
# print(
#    f"average agreement rate for predictions: {np.round(avg_prediction_agreement,2)}%"
# )
# print(
#    f"total agreement rate: {np.round(np.mean([avg_vote_agreement,avg_prediction_agreement]),2)}%"
# )
correct_vote_pct = (
    100 * sum([vote in correct_vote for vote in all_votes]) / len(all_votes)
)
print(f"correct rate vote percentage = {correct_vote_pct}%")
# correct_pred_pct = (
#    100 * sum([pred in correct_pred for pred in all_preds]) / len(all_preds)
# )
"""
print("All dates mentioned:")
for dates in all_dates:
    print(dates)

print("All metrics mentioned:")
for metrics in all_metrics:
    print(metrics)
"""

# Load a compact, high-quality model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get the actual FOMC statement embedding
actual_fomc_statement = fetch_fomc_statement(meeting_date_fomc)
actual_embedding = model.encode(actual_fomc_statement, convert_to_tensor=True)

# Store similarities
similarities = []

# Loop through LLM-generated reasonings
for reasoning in all_reasonings:
    reasoning_embedding = model.encode(reasoning, convert_to_tensor=True)
    sim = util.cos_sim(reasoning_embedding, actual_embedding)
    similarities.append(sim.item())

# Average similarity
avg_sim = sum(similarities) / len(similarities)

# Display results
# print(f"Average SentenceTransformer cosine similarity over {len(all_reasonings)} runs:")
print(f"Similarity: {avg_sim:.4f}")
