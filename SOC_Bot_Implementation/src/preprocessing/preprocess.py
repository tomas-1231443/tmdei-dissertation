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
    # """
    # Clean and extract the relevant incident narrative from a description cell.
    
    # Given the Description text format, I want to:
    #   1. Extract only the text in the "h2. Description" section.
    #   2. Remove formatting markers such as headings (h1., h2.), color tags ({color:...}), 
    #      code markers ({code}), and separator lines (----).
    #   3. Remove long JSON blocks that appear to be embedded (assumed to be noise for our purpose).
    #   4. Remove extraneous backslashes and extra whitespace.
      
    # Decisions:
    #   - Extracting "h2. Description": We assume that the relevant incident report details are
    #     contained in the section that starts with "h2. Description" and ends when the next "h2." heading starts.
    #   - Dropping headings and formatting tags: Markup like "h1.", "h2.", "{color:blue}", etc., are
    #     purely cosmetic and do not contribute to the semantic content needed for training.
    #   - Removing long JSON blocks: JSON structures in the text are detailed technical logs that are
    #     unlikely to help with determining priority/taxonomy and may add significant noise.
    
    # Parameters:
    #     text (str): The raw text from the "Description" column.
    
    # Returns:
    #     str: The cleaned text containing only the relevant incident narrative.
    # """

    # match = re.search(r'(?si)h2\.\s*description\s*(.*?)(?=h2\.)', text)
    # if match:
    #     text = match.group(1)
    # else:
    #     pass

    # text = text.lower()
    
    # text = re.sub(r'\{color:[^}]*\}', '', text)
    # text = re.sub(r'\{code\}', '', text)
    # text = re.sub(r'h\d+\.\s*', '', text)
    # text = re.sub(r'-{2,}', '', text)
    # text = re.sub(r'\\', '', text)
    
    # text = re.sub(r'\{.{50,}?\}', '', text)
    
    # text = re.sub(r"[\"']", "", text)
    
    # text = re.sub(r'(\b[\w\-\s]+\b)(\s+\1)+', r'\1', text)
    
    # text = re.sub(r'\s+', ' ', text).strip()
    
    # return text

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

def preprocess_bulk_alerts(df: pd.DataFrame) -> pd.DataFrame:
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
    required_columns = ["Description", "Priority", "Taxonomy"]
    
    df = df[required_columns]
    
    df = df.dropna(subset=required_columns)

    df = df[(df["Priority"] != "P4") & (df["Taxonomy"].str.lower() != "other")]
    
    tqdm.pandas(desc="Cleaning descriptions")
    df["Description"] = df["Description"].progress_apply(clean_text)
    
    df["Priority"] = df["Priority"].astype(str).str.strip()
    df["Taxonomy"] = df["Taxonomy"].astype(str).str.strip()
    
    return df
