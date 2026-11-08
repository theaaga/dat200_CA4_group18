import pandas as pd 
from sklearn.model_selection import StratifiedKFold

# Setup cross-validation
RANDOM_STATE = 42
SCORING = "f1_macro"
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def clean(df):
    df = df.copy()
    
    # Convert InternationalPlan and VoiceMailPlan to binary
    for col in ["InternationalPlan", "VoiceMailPlan"]:
        df[col] = df[col].map({"yes": 1, "no": 0})
        
    # Convert AreaCode to string
    df["AreaCode"] = df["AreaCode"].astype(str)
    
    # Double check that there are no missing values
    assert df.isnull().sum().sum() == 0
    
    return df

def load_data(train_path="trainCA4DAT200.csv", test_path="testCA4DAT200.csv"):
    train_df = clean(pd.read_csv(train_path))
    test_df = clean(pd.read_csv(test_path))
    test_ids = test_df["CustomerId"]
    
    X = train_df.drop(columns=["CustomerId", "Churn"])
    y = train_df["Churn"]
    X_test = test_df.drop(columns="CustomerId")
    return X, y, X_test, test_ids