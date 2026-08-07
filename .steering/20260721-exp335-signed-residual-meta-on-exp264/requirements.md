# 要件

## 依頼

- exp264の既存selector出力74列を置き換えず、候補ごとのsigned residual予測だけをadd-onlyする実験を設計確定する。
- steering、実験ディレクトリ、`KAGGLE_DIRECTION.md`のバックログ、`experiment_summary.md`を作成・更新する。
- 設計確定時点では実装コード、Jupytext source、学習済みmodel、Kaggle package、push、学習、推論、提出を作成・実行しない。
- 2026-07-21の追加依頼「exp335を実装してください」により、Stage Sの実装と0-booster preflight実装だけを承認済みへ変更する。Kaggle package/push、preflight実行、20 CPU booster学習、Stage D、推論、提出は引き続き未承認とする。
- その後の段階的な「実行してください」により、0-booster preflightと20 CPU booster Stage S学習を順に承認し、Kaggle CPU version 2 / 3で実行した。Stage Sはtechnical/score gateをPASSした。2026-07-22にStage D実装と固定15 GPU booster実行も承認され、Kaggle T4 version 2で完了した。固定tail guard FAIL後、2026-07-23の別overrideにより保存済みmodelのCPU inferenceとsubmit-checkだけを実行した。外部提出は実行していない。

## 背景と根拠

- corrected exp264 Stage C v6は、12候補に対する`pred_abs_error`と`p_within10`をstrict nestedで生成し、74列compact meta featureとして後段TVTモデルへ渡した。
- selector scoreは全foldでpriorより良く校正された一方、hard top-1はfixed候補より悪化した。したがってselectorはhard選択器ではなく、後段TVTモデルへ候補品質を伝えるmeta-feature生成器として扱う。
- corrected exp264 Stage D v3ではclean 273特徴だけのcontrol `10.476169`から、273 + compact 74 = 347特徴のadd-only `8.460811237612477`へ改善した。既存74列は有効な保存済みcontrolであり、削除や再学習を本実験に混ぜない。
- 既存2 objectiveは誤差の大きさを表すが、候補が真値に対して上側・下側のどちらへずれるかを直接表さない。候補別の`true_tvt - candidate_tvt`をfold-safeに予測し、方向情報を後段へ追加する価値を検証する。

## 制約

- Route: `ml_model`。PF/HMM/Beam候補は補助meta featureであり、最終TVTは後段LightGBMが生成する。
- 親実験と主control: `exp264_exp263_candidate_confidence_dual_selector` corrected Stage C v6 / Stage D v3。
- selector入力はexp264修正版のraw-test-safe 88特徴、12候補、候補順、2 legal domain、outer 5 × inner 4 split、seed 42を固定する。
- 既存の`pred_abs_error` / `p_within10` model、nested compact 74列、clean 273特徴はSHA固定された保存済み生成物を再利用し、再学習しない。
- 新しい教師は`selector_signed_residual = true_tvt - candidate_tvt`だけとし、LightGBM objectiveは`regression_l2`、1 configに固定する。label clip、Huber、quantile、candidate別objective、sample weight、target変換gridを混ぜない。
- 新規selector modelはstrict nested outer 5 × inner 4 = 20 CPU boostersだけとする。非nested Stage B、既存2 objective、候補生成、control selectorを再学習しない。
- 新規compactは23列に固定する。既存74列を変更・削除・再計算しない。
- 後段TVT variantはclean 273 + saved compact 74 + signed compact 23 = 370特徴、1 variant × 3 LightGBM configs × 5 folds = 15 GPU boostersとする。saved exp264 controlを使い、control再学習は0 boostersとする。
- Stage S実装、Kaggle package/push、0-booster preflight、20 CPU selector boosters、15 GPU TVT boostersは段階的承認を得て完了した。固定downstream gate FAILは維持し、後続の別overrideで保存済みmodel inferenceとsubmission file生成・submit-checkだけを例外実施する。
- `docs/06_reproducibility.md`に従い、fold、feature schema、入力、model、OOF、Kaggle packageのSHA契約を実装前から固定する。

## Selector教師契約

行`r`、候補`c`について、真のTVTを`t_r`、候補TVTを`v_r,c`とする。

`y_signed(r,c) = t_r - v_r,c`

- 正なら候補を深いTVT方向へ増加させる補正、負なら浅い方向へ減少させる補正を表す。
- labelはselector学習時だけ使い、後段へ渡すのはstrict nested OOF予測`pred_signed_residual`だけとする。
- `oracle_label`、`is_oracle`、actual error、actual rank、true TVT、targetをselector入力または後段特徴へ入れない。
- candidate unavailable / nonfinite行はexp264のavailability契約に従ってfail-closedまたは学習対象外とし、暗黙の0ラベルを作らない。

## Add-only 23特徴契約

### 候補別12列

- `selector__pred_signed_residual__<candidate_id>`をexp264の12候補順で12列作る。

### 既存top-1への方向注釈8列

2 legal domain × 2既存objective（`pred_abs_error` / `p_within10`）について次の2列を作る。

- `signed_residual_at_existing_top1`
- `signed_corrected_top1_minus_anchor = existing_top1_value + signed_residual_at_existing_top1 - last_known_tvt`

signed residual自体から新しいtop-1を定義しない。既存objectiveのtop-1 identityをそのまま使う。

### 分布3列

- 12候補の`pred_signed_residual_mean`
- 12候補の`pred_signed_residual_std`
- 12候補の`pred_signed_residual_range`

### 禁止する派生

- 全候補のcorrected TVT 12列追加
- signed residual最小絶対値によるhard top-1
- corrected TVTのsoftmax平均、Viterbi、hard switch、direct submission
- 23列以外のmargin、threshold、rolling、segment/well aggregateの同時追加

## 受け入れ基準

設計完了は次をすべて満たすこととする。

- steering、実験ディレクトリ、`KAGGLE_DIRECTION.md`、`experiment_summary.md`に同じ仮説、23列schema、実行量、control再学習0、禁止事項が記録されている。
- `config.yaml`でStage S、Stage D、CPU inferenceが承認・完了、全再実行flagが無効で、外部submissionが無効である。
- 検証済みcompact self-contained train / inference候補を正規notebookへ採用する。
- `make validate-exp EXP=exp335_signed_residual_meta_on_exp264`と実験文書監査が通る。

実行済みStage S selector gateは次をすべて満たした。

- 20/20 expected model、outer/inner well-disjoint、outer-valid truth非参照、全行/全候補finite prediction、candidate順、feature schema、partition SHAがPASSする。
- `pred_signed_residual`のpooled RMSEがcandidate別outer-train mean residual priorより改善する。
- 5 outer folds中4 folds以上で同priorよりRMSEが改善する。
- label / predictionの符号、`candidate_tvt + signed_residual = true_tvt`のlabel formula parity、保存modelとOOFのSHAがPASSする。
- Stage S gate不通過時は15 GPU downstream trainへ進まない。

Stage D version 2は15/15 modelsを完了し、pooled RMSE、4/5 folds、全scopeはPASSしたが、by-well delta p95 `+1.728657 ft`とworst-well delta `+10.238752 ft`が固定上限を超えた。clean273比promotion tailも悪化したためscientific-support / promotion gateをFAILとし、同一実験の救済なしでクローズした。その後のCPU inference overrideはこの判定を変更しない。

Stage D実行後のdownstream scientific support gateは次をすべて満たす。

- pooled OOF RMSEがsaved exp264 `8.460811237612477`から`0.03 ft`以上改善する。
- 5 folds中4 folds以上でsaved exp264以下になる。
- near / mid / 1000+ / hidden-like 2面がsaved exp264比で悪化しない。
- by-well RMSE delta p95がsaved exp264比で`<= 0.00 ft`、worst-well deltaが`<= +0.25 ft`になる。
- selector23列が非ゼロのgainを持ち、重要度が単一candidate IDだけへ極端に集中していないことをreadoutする。ただし重要度単独では採用しない。

train-side promotionはscientific supportに加え、clean 273 controlに対する既存exp264のworst-well / `+1/+3/+5 ft`悪化well数guardを悪化させず、事前に固定したclean-control guardをすべて満たす場合だけとする。gate不通過時はguardを緩和せず、signed residual objective/grid/特徴追加で同一実験を救済しない。

- deterministic anchorとして扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel versionを記録する。
- gzip生成物を比較する場合は、raw `.csv.gz` SHAではなくdecompressed content SHAを主証拠として記録する。

## 2026-07-23 CPU推論override

- ユーザーの「推論に進んでほしいです。GPU quotaがないのでCPU実行にできますか。」を、固定tail guard FAILを保持したまま保存済みmodelでCPU推論する明示承認として扱う。
- Kaggle CPU / internet offでcurrent test特徴を同一run内再生成し、Stage C 40 models、Stage S 20 models、Stage D 15 modelsをSHA検証して等重み推論する。学習boosterは0。
- 最終入力はtrainと同じclean273 + saved74 + signed23 = 370列、outer fold対応を維持する。
- `submission.csv`はsubmit-check用の推論成果物として生成するが、Kaggle competitionへの外部提出は承認されていない。
- scientific-support / promotion gateのFAILは変更せず、CPU推論をtrain-side採用の根拠にしない。
- CPU inference version 3はKaggle CPU / internet offで14,151 rows / 3 wellsを処理し、40/20/15 saved models、final 370特徴、formula/top-1 parityを検証して完了した。`submission.csv`はsampleとのheader・行数・ID順、重複、NaN/Infのsubmit-checkをPASSした。その後ユーザーがcode submissionを実施し、ref `54928806`はPublic LB `7.517`でCOMPLETEになった。agentによるsubmitは行っておらず、train-side非promote判断も変更しない。
