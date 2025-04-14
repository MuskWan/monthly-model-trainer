import pandas as pd
import matplotlib.pyplot as plt

def check_distribution_stability(pred_df, last_month_preds=None, label='奇点1_pred'):
    current_dist = pred_df[label].value_counts(normalize=True).sort_index()
    print(f"\n📊 当前月份 {label} 分布：\n{current_dist}")

    if last_month_preds is not None:
        last_dist = last_month_preds[label].value_counts(normalize=True).sort_index()
        combined = pd.DataFrame({
            'last_month': last_dist,
            'this_month': current_dist
        }).fillna(0)
        combined['diff'] = (combined['this_month'] - combined['last_month']).abs()
        print(f"\n🔍 {label} 分布变化：\n{combined}")
        print(f"📈 分布总变动程度（L1距离）：{round(combined['diff'].sum() * 100, 2)}%")

        combined[['last_month', 'this_month']].plot(kind='bar', figsize=(10, 5), title=f"Distribution Change: {label}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def high_value_prediction_check(pred_df, value_col='Total_Sales_USD', label='奇点1_pred', top_k=10):
    pred_df[value_col] = pd.to_numeric(pred_df[value_col], errors='coerce')
    top_df = pred_df.sort_values(by=value_col, ascending=False).head(top_k)
    print(f"\n💰 销售额Top {top_k}预测奇点分布（{label}）：")
    print(top_df[label].value_counts())

def compare_model_predictions(pred_df_current, pred_df_old, label='奇点1_pred'):
    merged = pred_df_current[['Product_URL_clean', label]].merge(
        pred_df_old[['Product_URL_clean', label]],
        on='Product_URL_clean',
        how='inner',
        suffixes=('_new', '_old')
    )
    diff_rate = (merged[f"{label}_new"] != merged[f"{label}_old"]).mean()
    print(f"\n⚠️ 模型预测差异率（{label}）：{round(diff_rate * 100, 2)}%")
    return diff_rate
