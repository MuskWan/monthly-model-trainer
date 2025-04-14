import pandas as pd

def compare_and_select_for_review(current_preds, last_preds, label='奇点1_pred', threshold=0.2, filename_prefix="待人工打标"):
    """
    比较当前模型和上一个模型的预测结果，识别预测不一致的样本，并输出人工打标建议文件。
    
    参数:
    - current_preds: 当前月份预测结果（含 Product_URL_clean, 奇点X_pred）
    - last_preds: 上一个月预测结果（结构相同）
    - label: 要比较的预测字段名（如 '奇点1_pred'）
    - threshold: 差异率超过此阈值才输出文件
    - filename_prefix: 导出文件的前缀名

    返回:
    - to_review_df: 待人工打标的数据 DataFrame（如果需要），否则为 None
    """
    # 合并当前和历史预测数据
    merged = current_preds[['Product_URL_clean', label]].merge(
        last_preds[['Product_URL_clean', label]],
        on='Product_URL_clean',
        how='inner',
        suffixes=('_now', '_prev')
    )

    # 筛选预测不一致的样本
    merged['inconsistent'] = merged[f"{label}_now"] != merged[f"{label}_prev"]
    inconsistent_df = merged[merged['inconsistent']]

    # 计算差异率
    diff_rate = inconsistent_df.shape[0] / merged.shape[0]
    print(f"\n📉 {label} 差异率为：{round(diff_rate * 100, 2)}%")

    # 如果高于阈值，导出不一致样本
    if diff_rate > threshold:
        print("📤 差异率高，导出供人工审核的数据样本...")
        to_review_df = current_preds[current_preds['Product_URL_clean'].isin(inconsistent_df['Product_URL_clean'])]
        out_name = f"{filename_prefix}_{label}.xlsx"
        to_review_df.to_excel(out_name, index=False)
        print(f"✅ 已保存至文件：{out_name}")
        return to_review_df
    else:
        print("✅ 差异率正常，无需人工复审。")
        return None
