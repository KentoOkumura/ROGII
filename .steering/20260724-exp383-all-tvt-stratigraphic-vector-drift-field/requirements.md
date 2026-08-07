# 要件

## 依頼

`exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` の
「各学習坑井のTVT driftをK=16の傾きへ分解し、近傍坑井から対象位置へ補間する」
物理モデルを、全outer-train坑井の正解TVTと6つのtrain-only地層面を教師として
全面的に作り替える。

目的はexp226の微修正ではなく、全TVTから高密度な地層ドリフトの絶対場と
2次元ベクトル場を復元し、対象坑井の全既知prefixで校正した物理pathを生成することにある。
物理モデル単独でPublic LB 6.5を目指すロードマップのP0とする。

今回はbacklog、steering、実験ディレクトリと設計契約だけを確定する。
実装、正規Notebookの作り替え、Kaggle package/push/run、推論、提出は行わない。

## 制約

- Route: `pf_beam`
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- outer-valid/testの正解TVTと生の6地層列はcandidate生成に使わない。
- outer-trainでは正解TVTの既知prefixだけでなく全行を教師として使う。
- 地層列は`ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA`に固定する。
- targetで使える列は`MD/X/Y/Z/TVT_input`に限定し、GR/typewellはexp385まで使わない。
- 公開されている3 test wellsはhidden testのサンプルとして扱い、設定選択に使わない。
- exp226の12,368 K16 segmentを再重み付けするのではなく、全TVTを覆う固定multiscale
  windowから絶対`S=TVT+Z`と方向微分を作る。
- targetの全既知prefixを縦bias校正に使い、未知suffixの正解TVTはlate scoring以外で読まない。
- donor support、surface uncertainty、orientation conditionが悪い位置はexp226 rateへ連続縮約する。
- LightGBM/CatBoost/XGBoost/NN、HMM、PF particle sampling、Beam search、GR likelihoodは含めない。
- 同一OOFを見たwindow、近傍数、bandwidth、ridge、縮約式、gateの救済gridは禁止する。
- 再現性は`docs/06_reproducibility.md`に従い、入力、fold、surface、donor catalog、
  field、predictionのschema/content SHAを記録する。

## 受け入れ基準

- steering 3文書、実験scaffold、`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`に同じdesign-only契約が記録されている。
- `KAGGLE_DIRECTION.md`で既存exp377より前のP0として記録され、exp384/385の先行条件になっている。
- outer 5-fold、773 wells、3,783,989 score rows、保存済みexp226 CV
  `9.427109596582213`を基準として固定している。
- Stage 0でtarget-freeなsurface/field/support/prefix availabilityを監査し、FAIL時はscoreを開かない。
- Stage 1はexp226比`1.0 ft`以上のpooled改善、4/5 folds、long/hidden-like改善を要求する。
- by-well tailは初回の科学的識別可能性を隠さないようreport-onlyとし、推論・提出判断では別途hard gateにする。
- 実装前のNotebookはtemplate scaffoldのままで、実行可能な正規実装と扱わない。
- deterministic anchorはrerunでlogical content SHA一致を確認するまで主張しない。
- gzip生成物はdecompressed content SHAを主証拠にする。

## 2026-07-24 実装承認追記

ユーザーの「exp383を実装してください」を、別名compact self-contained train候補と
fail-closed inference候補、専用contract testの実装承認として扱う。
既存正規Notebookの上書き、Kaggle package/push/run、inference、submissionは
この指示に含めず、引き続き別承認とする。

実装後も科学契約は変更せず、1 candidate / 5 reporting folds /
model・HMM・PF・Beam・booster各0 / exp226 control再実行0を維持する。

## 2026-07-24 実行承認追記

ユーザーの「実行してください」を、compact trainの正規Notebook採用、
Kaggle CPU package/push、16-well Stage 0 resource preflight、およびpreflight PASS後の
full 5-fold run承認として扱う。inferenceとsubmissionは含めない。

preflightまたはfull runでStage 0がFAILした場合はtruthを開かず停止する。
preflight PASS前にfull runへ切り替えず、正規Notebook・bootstrap config・kernel metadataの
mode一致を各push前に確認する。
