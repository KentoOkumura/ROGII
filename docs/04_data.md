# データ

実験テンプレートが読む標準データパス、ID 列、提出ターゲット列は `project.yml` を正とします。このファイルは公式データの内容とリスクの運用メモです。

## ファイル

| ファイル | 行数 | 列数 | 用途 | メモ |
| --- | ---: | ---: | --- | --- |
| `data/raw/sample_submission.csv` | 14,151 | 2 | 公開例の提出形式検証 | `id,tvt`。隠しテストでは Kaggle 実行環境の sample に従う。 |
| `train/{WELLNAME}__horizontal_well.csv` | well ごとに可変 | 13 | 学習用 horizontal well | 例 `000d7d20`: 5,278 行。 |
| `train/{WELLNAME}__typewell.csv` | well ごとに可変 | 3 | 学習用 typewell reference log | 例 `000d7d20`: 1,296 行。 |
| `train/{WELLNAME}.png` | - | - | 可視化 | well path と geological cross-section。 |
| `test/{WELLNAME}__horizontal_well.csv` | well ごとに可変 | 6 | 推論用 horizontal well | 公開例 `000d7d20`: 5,278 行。隠しテストで差し替え。 |
| `test/{WELLNAME}__typewell.csv` | well ごとに可変 | 3 | 推論用 typewell reference log | 隠しテストで差し替え。 |
| `AI_wellbore_geology_prediction_task_en.pptx` | - | - | 公式タスク説明 | 28.8 MB。必要時に取得する。 |

## スキーマ

- ID 列: 提出では `id`。形式は `{WELLNAME}_{row_index}`。
- ターゲット列: 学習データでは `TVT`、提出では `tvt`。
- グループ列: 明示列はないため、ファイル名から `well_id` を作る。
- 深度/順序列: `MD`。Measured Depth in ft。
- horizontal train columns: `MD`, `X`, `Y`, `Z`, `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`, `TVT`, `GR`, `TVT_input`
- horizontal test example columns: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`
- typewell columns: `TVT`, `GR`, `Geology`

## 分割

- Train: 773 wells。各 well に horizontal CSV、typewell CSV、PNG がある。
- Test: 公開ファイルは 3 wells の例のみ。公式説明では hidden test は約 200 wells で、Notebook 再実行時に差し替えられる。
- サンプル提出: 3 wells、14,151 行。well 別行数は `000d7d20`: 3,836、`00bbac68`: 6,014、`00e12e8b`: 4,301。
- 公式ファイル一覧: 2,327 files、合計 1,325,112,684 bytes。内訳は root 2 files、visible test 6 CSV、train 1,546 CSV + 773 PNG。

## EDA メモ

- `000d7d20` の horizontal train では `TVT_input` NaN が 3,836 行で、sample submission の同 well 行数と一致する。
- `GR` は欠損が多い。`000d7d20` horizontal では 5,278 行中 2,258 行が NaN。
- typewell の `Geology` も欠損があり、`000d7d20` typewell では 1,296 行中 299 行が NaN。
- `TVT_input` が非 NaN の既知区間は、軌道ごとの初期地質位置を強く示す。評価区間の外挿方法がベースラインの中心になる。

## データリスク

- リーク候補: 同一 well の row-level split、`TVT_input` NaN 区間より後ろの `TVT` 参照、train-only formation columns の使用、test 全体で fit した imputer/scaler。
- 重複: `id` は sample submission では重複なし。生成時に必ず確認する。
- 欠損値: `GR`、`TVT_input`、`Geology` に欠損がある。`TVT_input` の欠損は提出対象区間として意味を持つ。
- 分布シフト: 公開 `test/` は train 由来の 3 well 例で、hidden test とは well 数も分布も異なる可能性がある。
- 形式差: train horizontal にある formation columns は test example にないため、学習/推論の feature alignment を必ず検証する。
