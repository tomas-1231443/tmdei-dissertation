import pandas as pd
from src.models.model_training import train_false_positive_detector
from sklearn.ensemble import RandomForestClassifier
from src.config import EXCEL_COLUMNS

def test_train_false_positive_detector():
    n = 100
    # Create dummy numeric data for all columns except "Status"
    data = {}
    for col in EXCEL_COLUMNS:
        if col == "Status":
            continue
        data[col] = [1.0] * n
    # Create a binary label for "Status"
    data["Status"] = [0 if i % 2 == 0 else 1 for i in range(n)]
    
    df = pd.DataFrame(data)
    
    model = train_false_positive_detector(df, label_col="Status")
    assert isinstance(model, RandomForestClassifier)
    
    # Create a sample row (features only) to test prediction
    sample = df.drop(columns=["Status"]).iloc[0:1]
    prediction = model.predict(sample)
    assert prediction.shape[0] == 1
