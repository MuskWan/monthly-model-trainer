import os
import glob
import argparse
import pandas as pd
import joblib
from train_module import load_and_preprocess_data, train_model, predict_and_decode, evaluate_predictions
from active_learning_module import compare_and_select_for_review

DATA_DIR = "data"
MODEL_DIR = "model"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def get_all_emi_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "EMI_*.xlsx")))

def load_all_training_data(files):
    dataframes = [pd.read_excel(f) for f in files]
    return pd.concat(dataframes, ignore_index=True)

def auto_train_and_predict(latest_emi_file, latest_check_file=None, last_month_file=None):
    all_emi_files = get_all_emi_files(DATA_DIR)
    all_emi_df = load_all_training_data(all_emi_files)

    test_df = pd.read_excel(latest_emi_file)
    all_emi_df['Product_URL_clean'] = all_emi_df['Product_URL'].str.strip().str.lower()
    all_emi_df['text'] = all_emi_df['Brand_New'].astype(str) + " " + all_emi_df['Product_Title'].astype(str)

    test_df['Product_URL_clean'] = test_df['Product_URL'].str.strip().str.lower()
    test_df['text'] = test_df['Brand_New'].astype(str) + " " + test_df['Product_Title'].astype(str)

    model, label_encoders = train_model(all_emi_df)
    joblib.dump((model, label_encoders), os.path.join(MODEL_DIR, "latest_model.pkl"))

    predictions = predict_and_decode(model, test_df, label_encoders)

    month_tag = os.path.basename(latest_emi_file).split("_")[-1].replace(".xlsx", "")
    output_path = os.path.join(RESULT_DIR, f"predictions_{month_tag}.xlsx")
    predictions.to_excel(output_path, index=False)
    print(f"✅ 已保存预测结果至 {output_path}")

    if last_month_file:
        last_preds = pd.read_excel(last_month_file)
        print("\n🔁 启动主动学习筛选（预测不一致样本）...")
        for label in ['奇点1_pred', '奇点2_pred', '奇点3_pred']:
            compare_and_select_for_review(predictions, last_preds, label=label)

    if latest_check_file:
        check_df = pd.read_excel(latest_check_file)
        check_df['Product_URL_clean'] = check_df['Product_URL'].str.strip().str.lower()
        check_df = check_df.rename(columns={
            '奇点1': '奇点1_true', '奇点2': '奇点2_true', '奇点3': '奇点3_true'
        })
        results = evaluate_predictions(predictions, check_df)
        print("\n📊 有验证集，输出准确率：")
        print("🎯 奇点1准确率：", round(results['accuracy_q1'] * 100, 2), "%")
        print("🎯 奇点2准确率：", round(results['accuracy_q2'] * 100, 2), "%")
        print("🎯 奇点3准确率：", round(results['accuracy_q3'] * 100, 2), "%")
        print("🌐 全局准确率：", round(results['global_accuracy'] * 100, 2), "%")
        print("💰 销售额占比：", round(results['sales_usd_ratio'] * 100, 2), "%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, required=True, help="当前月份（如 202504）")
    parser.add_argument("--last", type=str, default=None, help="上个月份（默认=前一个月）")
    parser.add_argument("--check", action="store_true", help="是否包含验证集")
    args = parser.parse_args()

    current_month = args.month
    last_month = args.last if args.last else str(int(current_month) - 1)

    emi_file = os.path.join(DATA_DIR, f"EMI_{current_month}.xlsx")
    check_file = os.path.join(DATA_DIR, f"Check_{current_month}.xlsx") if args.check else None
    last_pred_file = os.path.join(RESULT_DIR, f"predictions_{last_month}.xlsx")

    auto_train_and_predict(
        latest_emi_file=emi_file,
        latest_check_file=check_file if check_file and os.path.exists(check_file) else None,
        last_month_file=last_pred_file if os.path.exists(last_pred_file) else None
    )
