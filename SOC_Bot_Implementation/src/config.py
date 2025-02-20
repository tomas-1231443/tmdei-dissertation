# This file holds any constants or configuration items.

VERBOSE = False

DEFAULT_MODEL_PATH = "src/models/trained_model.joblib"

EXCEL_COLUMNS = [
    "Issue ID",
    "Issue Type",
    "Status",
    "Description",
    "Custom field (Alert Technology)",
    "Custom field (Incident Description)",
    "Custom field (Incident Resolution 1)",
    "Custom field (Request Type)",
    "Custom field (Source)",
    "Custom field (Source Alert Rule Name)",
    "Custom field (Source_Alert_Id)",
    "Custom field (Taxonomy)",
    "Priority"
]

# Example placeholders for possible future expansions
DEFAULT_EXCEL_PATH = "data/sample_alerts.xlsx"
