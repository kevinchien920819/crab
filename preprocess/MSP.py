import polars as pl
import pickle
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def load_transcripts(trans_dir: str) -> dict:
    """
    讀取 Transcripts 資料夾下的所有 .txt 檔案，回傳 {filename: sentence} 的字典。
    """
    trans_path = Path(trans_dir)
    sentence_map = {}
    if not trans_path.exists():
        print(f"警告: 找不到逐字稿目錄 {trans_dir}")
        return sentence_map

    print(f"正在從 {trans_dir} 讀取逐字稿內容...")
    txt_files = list(trans_path.glob("*.txt"))
    if not txt_files:
        print(f"警告: 在 {trans_dir} 中找不到任何 .txt 檔案")
        return sentence_map
    
    for txt_file in tqdm(txt_files, desc="讀取逐字稿"):
        # stem 取得不含副檔名的檔名
        filename = txt_file.stem 
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                sentence_map[filename] = content
        except:
            try:
                with open(txt_file, "r", encoding="latin-1") as f:
                    content = f.read().strip()
                    sentence_map[filename] = content
            except:
                pass
    return sentence_map

def preprocess_combined(label_csv: str, trans_dir: str, output_pkl: str):
    """
    讀取標籤 CSV 並合併對應的逐字稿內容，最後存成一個 pkl 檔案。
    """
    if not os.path.exists(label_csv):
        print(f"錯誤: 找不到標籤檔案 {label_csv}")
        return

    # 1. 先讀取所有的逐字稿到記憶體
    sentence_map = load_transcripts(trans_dir)
    
    # 2. 讀取標籤 CSV
    print(f"正在讀取標籤檔: {label_csv}...")
    try:
        df = pl.read_csv(label_csv)
    except Exception as e:
        print(f"讀取 CSV 失敗: {e}")
        return
    
    # 標準化欄位名稱
    target_columns = ["FileName", "EmoClass", "Split_Set"]
    for target in target_columns:
        if target not in df.columns:
            for actual in df.columns:
                if target.lower() == actual.lower():
                    df = df.rename({actual: target})
                    print(f"  自動匹配欄位: {actual} -> {target}")
                    break

    # 3. 合併資料並建立最終列表
    print("正在合併標籤與逐字稿...")
    final_data = []
    
    # 轉換為 dicts 進行處理
    rows = df.to_dicts()
    
    # 定義 Split_Set 的映射表
    split_map = {
        "Train": "train",
        "Development": "dev",
        "Test1": "test1",
        "Test2": "test2"
    }
    
    for row in tqdm(rows, desc="合併進度"):
        # 取得不含副檔名的 key 以便與逐字稿 map 對接
        fn_key = Path(row["FileName"]).stem
        
        # 新增 Sentence 欄位，若找不到逐字稿則設為空字串或 None
        row["Sentence"] = sentence_map.get(fn_key, "")
        
        # 標準化 Split_Set 的值
        if "Split_Set" in row:
            row["Split_Set"] = split_map.get(row["Split_Set"], row["Split_Set"])
        
        final_data.append(row)

    # 4. 儲存結果
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(final_data, f)
        
    print(f"\n預處理完成！")
    print(f"輸出檔案: {output_pkl}")
    print(f"樣本總數: {len(final_data)}")
    if final_data:
        print(f"包含欄位: {list(final_data[0].keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSP-Podcast 標籤與逐字稿整合工具")
    parser.add_argument("--label_in", "-l", type=str, default="dataset/MSP_Podcast/Labels/labels_consensus.csv", help="標籤 CSV 路徑")
    parser.add_argument("--trans_in", "-t", type=str, default="dataset/MSP_Podcast/Transcripts", help="逐字稿目錄路徑")
    parser.add_argument("--output", "-o", type=str, default="dataset/MSP_Podcast/MSP_combined.pkl", help="輸出的整合 pkl 路徑")
    
    args = parser.parse_args()
    
    preprocess_combined(args.label_in, args.trans_in, args.output)
