# This file holds any constants or configuration items.

VERBOSE = False

DEFAULT_MODEL_DIR = "src/models/"
DEFAULT_EXCEL_PATH = "data/historical_alerts.csv"
DEFAULT_CLEANED_ALERTS_PATH = "data/cleaned_alerts.csv"

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
