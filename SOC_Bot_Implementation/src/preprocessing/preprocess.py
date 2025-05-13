import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Union
from tqdm import tqdm
from src.logger import with_logger

from src.config import EXCEL_COLUMNS

# TODO: Further down the line, integrate a pipeline approach (scikit-learn Pipeline)
# TODO: Add advanced text cleaning or normalization if needed

@with_logger
def clean_text(text: str, *, logger) -> str:
    
    # 1. Extract content after "h2. Description"
    match_desc = re.search(r'(?si)h2\.\s*description\s*(.*?)(?=h2\.)', text)
    if not match_desc:
        cleaned = text

        # Remove "h2. Taxonomy/Taxonomia:" and all text after it.
        cleaned = re.split(r'(?si)h2\.\s*taxonomy\/taxonomia:?', cleaned)[0]
        
        # Remove "h2. References:" and all text after it.
        cleaned = re.split(r'(?si)h2\.\s*references:?', cleaned)[0]

        cleaned = re.sub(r'\#[0-9]+', '', cleaned)

        # Remove color tags (allowing spaces around the colon) and code markers.
        cleaned = re.sub(r'\{color\s*:?\s*[^}]*\}', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\{code\}', '', cleaned, flags=re.IGNORECASE)
        
        # Remove heading markers like "h1.", "h2.", separator lines (----) and backslashes.
        cleaned = re.sub(r'h\d+\.\s*', '', cleaned)
        cleaned = re.sub(r'-{2,}', '', cleaned)
        cleaned = re.sub(r'\\', '', cleaned)
        
        # Remove HTML tags.
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Remove extraneous formatting characters.
        cleaned = re.sub(r'[\*\{\}\[\]\(\)]', '', cleaned)
        
        # Remove IBM SOAR Link: strings.
        cleaned = re.sub(r'IBM SOAR Link:\s*\S+', '', cleaned, flags=re.IGNORECASE)
        
        # Remove URLs.
        cleaned = re.sub(r'http[s]?://\S+', '', cleaned)
        
        # Collapse extra whitespace.
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    extracted = match_desc.group(1)

    # 2. Remove "/descrição"
    extracted = re.sub(r'/descri[cç]ão', '', extracted, flags=re.IGNORECASE)
    
    # 3. Remove formatting markers
    extracted = re.sub(r'\{color:[^}]*\}', '', extracted)  # remove color tags
    extracted = re.sub(r'\{code\}', '', extracted)          # remove code markers
    extracted = re.sub(r'h\d+\.\s*', '', extracted)          # remove any heading markers like h1., h2.
    extracted = re.sub(r'-{2,}', '', extracted)              # remove separator lines (----)
    extracted = re.sub(r'\\', '', extracted)                 # remove backslashes
    extracted = re.sub(r'\*\*', '', extracted)               # remove asterisks
    extracted = re.sub(r'\[|\]', '', extracted)              # remove square brackets
    
    # 4. Remove trailing parts starting with "action:" or "related evidences:" (case insensitive)
    extracted = re.split(r'(?si)(action:|related evidences:)', extracted)[0]
    
    # 5. Look for "rule name:" (case insensitive)
    parts = re.split(r'(?si)rule name:\s*', extracted, maxsplit=1)
    if len(parts) < 2:
        return extracted.strip()
    
    narrative = parts[0].strip()
    rule_info = parts[1].strip()
    
    # Capture the rule name: take text up to the first newline or an asterisk, if present.
    rule = re.split(r'[\n\*]', rule_info)[0].strip()
    
    # 6. Format the final output as "<Rule Name> - <Narrative>"
    final_text = f"{rule} - {narrative}"
    
    # 7. Remove extra whitespace.
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    
    return final_text

@with_logger
def preprocess_bulk_alerts(df: pd.DataFrame, *, logger) -> pd.DataFrame:
    """
    Preprocesses the DataFrame of alerts for training an AI model that predicts
    'Priority' and 'Taxonomy' based on the raw text of the alert in the 'Description' column.
    
    Decisions made:
      - Keep "Description": This column contains the raw text input necessary for model training.
      - Keep "Priority": This is one of the target labels (the model will predict this).
      - Keep "Taxonomy": This is the other target label.
      - Drop all other columns: They include identifiers (e.g., "Issue id"), metadata (e.g., "Created", "Updated"), 
        or other details that do not contribute to the model's objective and might introduce noise or risk target leakage.
      - Drop rows with missing values: To ensure data consistency, we remove rows missing any of the essential columns.
      - Clean the 'Description' column: We reduce noise by lowercasing text, removing punctuation, and stripping extra spaces.
    """
    
    df = df[(df['Source'] != 'Other') & (df['Source Alert Rule Name'].notna())]

    required_columns = ["Description", "Priority", "Taxonomy"]
    
    df = df[required_columns]
    
    df = df.dropna(subset=required_columns)

    df = df[(df["Priority"] != "P4") & (df["Taxonomy"].str.lower() != "other")]

    # 5. Random oversampling by Taxonomy
    #    a) Identify the largest category size
    taxonomy_counts = df["Taxonomy"].value_counts()
    logger.debug(f"Taxonomy counts: {taxonomy_counts}")
    max_len = taxonomy_counts.max()
    logger.info(f"Max length: {max_len}")

    #    b) Oversample smaller categories to match max_len
    oversampled_dfs = []
    for tax, count in taxonomy_counts.items():
        sub_df = df[df["Taxonomy"] == tax]
        if count < max_len:
            # Number of additional samples needed
            diff = max_len - count
            # Randomly sample 'diff' rows with replacement
            extra_rows = sub_df.sample(n=diff, replace=True, random_state=42)
            sub_df = pd.concat([sub_df, extra_rows], ignore_index=True)
        oversampled_dfs.append(sub_df)

    #    c) Merge all oversampled data back
    df = pd.concat(oversampled_dfs, ignore_index=True)

    logger.debug(f"Oversampled taxonomy counts: {df['Taxonomy'].value_counts()}")

        # Extract the Critical Asset flag into its own column:
    def extract_critical(text: str) -> int:
        m = re.search(r'Related with Critical Asset:\s*\{color:red\}(True|False)', text, flags=re.IGNORECASE)
        return 1 if m and m.group(1).lower() == 'true' else 0

    tqdm.pandas(desc="Extracting CriticalAsset flag")
    df["CriticalAsset"] = df["Description"].progress_apply(extract_critical)
    
    tqdm.pandas(desc="Cleaning descriptions")
    df["Description"] = df["Description"].progress_apply(clean_text)
    
    df["Priority"] = df["Priority"].astype(str).str.strip()
    df["Taxonomy"] = df["Taxonomy"].astype(str).str.strip()
    
    return df