# コンペ概要

このファイルは、コンペ全体の運用メモです。機械可読なコンペ設定の正は `project.yml` とし、ここには公式情報の解釈、制約、未解決事項を残します。公式評価ページからの抜粋と出典は `docs/official/evaluation.md` に残します。

## 基本情報

- コンペ名: ROGII - Wellbore Geology Prediction
- URL: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- Slug: `rogii-wellbore-geology-prediction`
- 主催: ROGII
- カテゴリ/賞金: Featured / $50,000
- 目的: 水平井掘削中に遭遇する地質位置を機械学習で予測し、well placement の精度向上に寄与する。
- 予測対象: 各 horizontal well の evaluation zone における `TVT` / `tvt`。
- 提出形式: Notebook-only code competition。Notebook 実行で `submission.csv` を生成する。
- スケジュール:
  - 2026-05-05: Start Date
  - 2026-07-29 23:59 UTC: Entry Deadline / Team Merger Deadline
  - 2026-08-05 23:59 UTC: Final Submission Deadline

## 重要ポイント

- 水平井ごとに `{WELLNAME}__horizontal_well.csv` と `{WELLNAME}__typewell.csv` が対応する。
- `TVT_input` は既知区間の `TVT` コピーで、evaluation zone は NaN になる。提出対象はこの NaN 区間に対応する。
- train には 773 well があり、各 well に horizontal CSV、typewell CSV、可視化 PNG がある。
- 公開 `test/` は train 由来の 3 well 例のみ。実提出時は約 200 well の隠しテストに差し替えられる。
- train の `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` は公式説明で Training only とされている。推論時に存在しない前提で扱う。
- 2026-05-27 取得時点の Public LB best は 7.973。CV と LB の対応はまだ未確認。

## 制約

- インターネット: Kaggle Notebook 提出では disabled。
- GPU/CPU: CPU Notebook と GPU Notebook のどちらも利用可。
- 実行時間: CPU/GPU ともに 9 hours 以内。
- 外部データ: Freely & publicly available external data は許可。pre-trained models も許可。
- 提出回数: 1 日 5 回。
- チーム/マージ規則: 最大チームサイズ 5。チームマージ締切は 2026-07-29 23:59 UTC。
- コード共有: team 外の private sharing は不可。公開する場合は Kaggle の competition forum/notebook 上で全参加者に公開する。
- 手作業ラベル: validation/test records の hand labeling や human prediction は不可。

## 提出

- ファイル: `submission.csv`
- ID 列: `id`
- ターゲット列: `tvt`
- 必要行数: `data/raw/sample_submission.csv` と同じ行数。公開例では 14,151 行、隠しテストでは Kaggle 実行環境の sample に従う。
- ID 形式: `{WELLNAME}_{row_index}`。例: `000d7d20_1442`
- 公式出典: Kaggle `Evaluation`、`data-description`、`Code Requirements` ページ。取得日: 2026-05-27。

## 未解決の質問

- Public/private split の well 数、地質分布、地域差は非公開。
- train-only 地層面列を代替する推論可能特徴をどう作るか。
- typewell の `Geology` ラベルをどこまで使うと隠しテストに安定するか。
- well holdout CV と Public LB の相関。
