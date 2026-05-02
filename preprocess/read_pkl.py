import pickle
import polars as pl
import argparse
import os

def read_and_show_pkl(pkl_path: str, num_rows_per_split: int = 3):
    """
    讀取 pkl 檔案，並統計 Split_Set 分布，同時顯示各類別的範例。
    """
    if not os.path.exists(pkl_path):
        print(f"錯誤: 找不到檔案 {pkl_path}")
        return

    print(f"正在從 {pkl_path} 讀取數據...")
    
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"讀取 pkl 失敗: {e}")
        return
    
    total_count = len(data)
    print(f"讀取成功！總筆數: {total_count}")

    if isinstance(data, list):
        # 將整份 list 轉換為 Polars DataFrame 以利統計
        df = pl.DataFrame(data)
        
        # 1. 顯示整體 Split_Set 統計
        if "Split_Set" in df.columns:
            print("\n=== [資料分布統計] ===")
            stats = (
                df.group_by("Split_Set")
                .agg(pl.count().alias("Count"))
                .with_columns((pl.col("Count") / total_count * 100).round(2).alias("Percentage (%)"))
                .sort("Count", descending=True)
            )
            print(stats)
            
            # 2. 從每個 Split_Set 中各抽樣幾筆來顯示
            print(f"\n=== [各組數據抽樣範例 (每組顯示 {num_rows_per_split} 筆)] ===")
            # 遍歷所有的 split 類別
            unique_splits = df["Split_Set"].unique().sort().to_list()
            
            sample_dfs = []
            for split in unique_splits:
                # 取得該 split 的子集並抽樣
                sub_df = df.filter(pl.col("Split_Set") == split).head(num_rows_per_split)
                sample_dfs.append(sub_df)
            
            # 合併抽樣結果顯示
            final_sample = pl.concat(sample_dfs)
            # 調整顯示設定：讓字串顯示完整一點
            with pl.Config(fmt_str_lengths=50, tbl_rows=50):
                print(final_sample)
        else:
            print("\n[提示] 找不到 'Split_Set' 欄位，僅顯示前幾筆數據：")
            print(df.head(num_rows_per_split * 4))

    elif isinstance(data, dict):
        print("偵測到字典格式，顯示前 10 筆 Key-Value...")
        for i, (k, v) in enumerate(list(data.items())[:10]):
            print(f"{k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="進階讀取 MSP-Podcast pkl 工具")
    parser.add_argument("--input", "-i", type=str, default="dataset/MSP_Podcast/MSP_combined.pkl", help="pkl 路徑")
    parser.add_argument("--n", "-n", type=int, default=3, help="每個 Split 顯示幾筆範例")
    
    args = parser.parse_args()
    read_and_show_pkl(args.input, args.n)