# train_module.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(emi_path, test_path, check_path):
    emi_df = pd.read_excel(emi_path)
    test_df = pd.read_excel(test_path)
    check_df = pd.read_excel(check_path)

    for df in [emi_df, test_df, check_df]:
        df['Product_URL_clean'] = df['Product_URL'].str.strip().str.lower()

    emi_df['text'] = emi_df['Brand_New'].astype(str) + " " + emi_df['Product_Title'].astype(str)
    test_df['text'] = test_df['Brand_New'].astype(str) + " " + test_df['Product_Title'].astype(str)

    check_df = check_df.rename(columns={
        '奇点1': '奇点1_true', '奇点2': '奇点2_true', '奇点3': '奇点3_true'
    })

    return emi_df, test_df, check_df

def train_model(emi_df):
    X_train = emi_df['text']
    y_train = emi_df[['奇点1', '奇点2', '奇点3']]

    label_encoders = {col: LabelEncoder().fit(y_train[col]) for col in y_train.columns}
    y_encoded = np.column_stack([label_encoders[col].transform(y_train[col]) for col in y_train.columns])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
    ])

    pipeline.fit(X_train, y_encoded)
    return pipeline, label_encoders

def predict_and_decode(model, test_df, label_encoders):
    X_test = test_df['text']
    y_pred_encoded = model.predict(X_test)

    decoded_preds = pd.DataFrame({
        col + '_pred': label_encoders[col].inverse_transform(y_pred_encoded[:, i])
        for i, col in enumerate(['奇点1', '奇点2', '奇点3'])
    })

    return pd.concat([test_df.reset_index(drop=True), decoded_preds], axis=1)

def evaluate_predictions(pred_df, check_df):
    merged = pred_df.merge(
        check_df[['Product_URL_clean', '奇点1_true', '奇点2_true', '奇点3_true', 'Total_Sales_USD']],
        on='Product_URL_clean', how='left')

    acc_q1 = (merged['奇点1_pred'] == merged['奇点1_true']).mean()
    acc_q2 = (merged['奇点2_pred'] == merged['奇点2_true']).mean()
    acc_q3 = (merged['奇点3_pred'] == merged['奇点3_true']).mean()
    all_correct = (
        (merged['奇点1_pred'] == merged['奇点1_true']) &
        (merged['奇点2_pred'] == merged['奇点2_true']) &
        (merged['奇点3_pred'] == merged['奇点3_true'])
    )
    global_acc = all_correct.mean()

    usd_col = [col for col in merged.columns if 'Total_Sales_USD' in col][0]
    merged[usd_col] = pd.to_numeric(merged[usd_col], errors='coerce')
    correct_usd = merged.loc[all_correct, usd_col].sum()
    total_usd = merged[usd_col].sum()
    sales_ratio = correct_usd / total_usd if total_usd > 0 else 0

    return {
        'accuracy_q1': acc_q1,
        'accuracy_q2': acc_q2,
        'accuracy_q3': acc_q3,
        'global_accuracy': global_acc,
        'sales_usd_ratio': sales_ratio,
        'merged_df': merged
    }
