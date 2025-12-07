import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# Storage and Set Up

# Where Files are Stored
root = r"C:\Users\alyss\PycharmProjects\Semester Project\.venv"

# File Names
FYE_file = "UNO_Student_Experience_Survey_1.csv"
NFYE_file = "UNO_Student_Experience_Survey_2.csv"

FYE_path = os.path.join(root, FYE_file)
NFYE_path = os.path.join(root, NFYE_file)


#Load and Clean
def load_csv(path, group_label, min_progress=0):
    df = pd.read_csv(path)

    # Drop the first two rows:
    #   row 0 = question text
    #   row 1 = ImportId metadata
    df = df.iloc[2:, :].reset_index(drop=True)

    # Basic type cleaning
    if "Progress" in df.columns:
        df["Progress"] = pd.to_numeric(df["Progress"], errors="coerce")
    if "Duration (in seconds)" in df.columns:
        df["Duration (in seconds)"] = pd.to_numeric(
            df["Duration (in seconds)"], errors="coerce"
        )

    # Filter to finished responses only
    if "Finished" in df.columns:
        df = df[df["Finished"] == "True"]

    # Min progress threshold (if min_progress is not None)
    if "Progress" in df.columns and min_progress is not None:
        df = df[df["Progress"] >= min_progress]

    # Add group label: "FYE" or "NonFYE"
    df["group"] = group_label

    return df

#Recode ordinal and binary scales to numeric
# add map for demographics
def recode_scales(df):
    agree_map = {
        "Strongly disagree.": 1,
        "Disagree.": 2,
        "Neutral.": 3,
        "Agree.": 4,
        "Strongly agree.": 5
    }
    agree_set = set(agree_map.keys())

    connected_map = {
        "Not connected at all.": 1,
        "Neutral.": 2,
        "Somewhat connected.": 3,
        "Connected.": 4
    }
    connected_set = set(connected_map.keys())

    freq_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4+": 4
    }
    freq_set = set(freq_map.keys())

    yesno_map = {
        "No.": 0,
        "Yes.": 1,
        "No": 0,
        "Yes": 1,
        "Prefer not to say": np.nan,
        "Prefer not to say.": np.nan
    }
    yesno_set = set(k for k in yesno_map.keys() if yesno_map[k] in [0, 1])

    gpa_map = {
        "2.5 or lower": 1,
        "2.5-3.0": 2,
        "3.0-3.25": 3,
        "3.25-3.5": 4,
        "3.5-3.75": 5,
        "3.75+": 6
    }
    gpa_set = set(gpa_map.keys())

    hours_map = {
        "5 or fewer hours.": 1,
        "6 - 10 hours.": 2,
        "11 - 15 hours.": 3,
        "16 - 20 hours.": 4,
        "21 - 25 hours.": 5,
        "Greater than 25 hours.": 6
    }
    hours_set = set(hours_map.keys())

    change_map = {
        # Academic performance
        "My grades have improved on average.": 3,
        "My grades have stayed the same on average.": 2,
        "My grades have worsened on average.": 1,

        # Time management skills
        "My time management skills have improved.": 3,
        "My time management skills have stayed the same.": 2,
        "My time management skills have worsened.": 1,

        # Study habits
        "I study more than before.": 3,
        "My study habits have not changed.": 2,
        "I study less than before.": 1
    }
    change_set = set(change_map.keys())

    for col in df.columns:
        # Only bother recoding question columns
        if not col.startswith("Q"):
            continue

        vals = set(df[col].dropna().unique())
        if not vals:
            continue

        if vals.issubset(agree_set):
            new_col = col + "_num"
            df[new_col] = df[col].map(agree_map).astype("float")

        elif vals.issubset(connected_set):
            new_col = col + "_num"
            df[new_col] = df[col].map(connected_map).astype("float")

        elif vals.issubset(freq_set):
            new_col = col + "_num"
            df[new_col] = df[col].map(freq_map).astype("float")

        elif vals.issubset(yesno_set):
            new_col = col + "_num"
            df[new_col] = df[col].map(yesno_map).astype("float")

        elif vals.issubset(gpa_set):
            # "What is your current GPA?"
            new_col = col + "_num"
            df[new_col] = df[col].map(gpa_map).astype("float")

        elif vals.issubset(hours_set):
            # "Hours per week on academic work outside of class"
            new_col = col + "_num"
            df[new_col] = df[col].map(hours_map).astype("float")

        elif vals.issubset(change_set):
            # Change questions: grades, time management, study habits
            new_col = col + "_num"
            df[new_col] = df[col].map(change_map).astype("float")

    return df

#Score financial literacy items Q37–Q41
def score_financial_literacy(df):
    def binary_score(series, correct_answer):
        return np.where(
            series.isna(), np.nan,
            np.where(series == correct_answer, 1, 0)
        )

    if "Q37" in df.columns:
        df["Q37_correct"] = binary_score(df["Q37"], "More than $102.")
    if "Q38" in df.columns:
        df["Q38_correct"] = binary_score(df["Q38"], "Less than today.")
    if "Q39" in df.columns:
        df["Q39_correct"] = binary_score(df["Q39"], "They will fall.")
    if "Q40" in df.columns:
        df["Q40_correct"] = binary_score(df["Q40"], "True.")
    if "Q41" in df.columns:
        df["Q41_correct"] = binary_score(df["Q41"], "False.")

    fin_cols = [c for c in df.columns if c.endswith("_correct")]
    if fin_cols:
        df["FinLit_Total"] = df[fin_cols].sum(axis=1)

    return df

#Master function - prepare combined dataset for analysis
def prepare_analysis_dataset(FYE_path, NFYE_path, min_progress=0):
    fye = load_csv(FYE_path, group_label="FYE", min_progress=min_progress)
    nfye = load_csv(NFYE_path, group_label="NonFYE", min_progress=min_progress)

    # Make sure both have same columns
    common_cols = fye.columns.intersection(nfye.columns)
    fye = fye[common_cols].copy()
    nfye = nfye[common_cols].copy()

    # Recode scales and score financial literacy, separately
    fye = recode_scales(fye)
    nfye = recode_scales(nfye)

    fye = score_financial_literacy(fye)
    nfye = score_financial_literacy(nfye)

    combined = pd.concat([fye, nfye], ignore_index=True)

    return combined



# Mann-Whitney Test Analysis
def run_mann_whitney(combined, variables=None, output_csv=None):
    if variables is None:
        variables = [
            c for c in combined.columns
            if c.endswith("_num") or c.endswith("_correct") or c == "FinLit_Total"
        ]

    results = []

    for col in variables:
        # Extract values for each group
        fye_vals = combined.loc[combined["group"] == "FYE", col].dropna()
        nonfye_vals = combined.loc[combined["group"] == "NonFYE", col].dropna()

        # Skip if no valid response for question like if one group did not answer
        if len(fye_vals) == 0 or len(nonfye_vals) == 0:
            continue

        # Mann–Whitney test (two-sided)
        u_stat, p_val = mannwhitneyu(fye_vals, nonfye_vals, alternative="two-sided")

        results.append({
            "Variable": col,
            "U_stat": u_stat,
            "p_value": p_val,
            "FYE_mean": fye_vals.mean(),
            "NonFYE_mean": nonfye_vals.mean(),
            "FYE_median": fye_vals.median(),
            "NonFYE_median": nonfye_vals.median(),
            "FYE_n": len(fye_vals),
            "NonFYE_n": len(nonfye_vals)
        })

    results_df = pd.DataFrame(results)

    # Sort by p-value
    if not results_df.empty:
        results_df = results_df.sort_values(by="p_value")

    if output_csv is not None:
        results_df.to_csv(output_csv, index=False)

    return results_df


# Print Results

if __name__ == "__main__":
    # 1. Prepare combined dataset
    combined = prepare_analysis_dataset(FYE_path, NFYE_path, min_progress=0)

    # 2. Run Mann–Whitney tests on all numeric question variables
    results_df = run_mann_whitney(
        combined,
        variables=None,  # auto-detect numeric variables
        output_csv=os.path.join(root, "mann_whitney_results.csv")
    )

    # 3. Print results to console
    if results_df.empty:
        print("No variables had data in both groups; check your dataset.")
    else:
        # Show a readable table in console
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 120)
        print(results_df)
