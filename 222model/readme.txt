[使用說明]
1. 先在colab上程式把各縣市的train, test抓下來 (有包含raw_time)，確保./data中有資料
2. 執行 tuning_use_better_validation.py (或其他相似名字程式): 計算出找參數的history
3. 用 ipynb跑後續: 讀取history, 重新訓練調參數模型和為調參數模型，並做test計算分數