# Kaggle Titanic: Machine Learning from Disaster

最初の Kaggle 定番コンペ「Titanic - Machine Learning from Disaster」に取り組んだ内容をまとめたリポジトリです。  
EDA → 欠損補完 → 特徴量作成 → 各種モデル比較 → 提出の流れを Jupyter Notebook で整理しています。

- タスク: 乗客の生存可否を二値分類
- 指標: Accuracy（Public/Private LB の比率 0.5/0.5）

## ディレクトリ構成

```
kaggle-titanic-survival-prediction/
├─ README.md
├─ data/                 # Kaggle 配布 train/test（手動配置）
├─ age_data/             # 年齢推定用補助データ
├─ cabin_data/           # Cabin 補完用データ
├─ now_titanic.ipynb     # メイン Notebook（EDA + モデル）
├─ now_titanic_age.ipynb # 年齢補完ロジック
├─ now_titanic_Cabin.ipynb
├─ now_titani_an.ipynb   # 追加解析
└─ submit/               # 提出履歴 CSV（19+ 回）
```

## セットアップ

1. Kaggle から `train.csv`, `test.csv`, `gender_submission.csv` を `data/` に配置
2. `jupyter lab` で `now_titanic.ipynb` を開く
3. 年齢補完・Cabin 推定が必要な場合は専用 Notebook を先に実行
4. 予測結果は `submit/` に書き出されます

## 特徴量エンジニアリング

- `Name` から敬称 (`Mr`, `Mrs`, `Master`, `Dr`) を抽出し、年齢や家族構成と組み合わせ
- `FamilySize = SibSp + Parch + 1` を導入し、大家族/単身の影響を確認
- `Ticket` の頭 1〜3 文字をカテゴリ化し、登場位置の proxy として活用
- Cabin 欠損に対してデッキ（A〜G）単位で補完を試み、`IsCabinMissing` フラグを追加
- 年齢は `RandomForestRegressor` を用いた回帰補完と中央値補完の比較を実施

## モデルと結果

| モデル | CV / LB 備考 |
| --- | --- |
| Logistic Regression | ベースライン, 交差検証 0.78 前後 |
| RandomForest | Hyperopt で深さ/本数を調整, Public LB 0.80台 |
| XGBoost / LightGBM | 学習率 0.05, 200 trees 前後で微改善 |
| Voting/Stacking | LR + RF + XGB をアンサンブル, 安定して 0.80+ を再現 |

Public LB では `submission_21.csv` 近辺が自己ベストとなっています（詳細は `submit/` 内の CSV と Kaggle 履歴を参照）。

## 今後の予定

- SHAP など解釈手法を導入し、特徴量設計を振り返る
- CatBoost / TabNet など追加モデルの導入
- 交差検証を StratifiedKFold + target leakage 対策付きで再構築
# kaggle-titanic-survival-prediction
