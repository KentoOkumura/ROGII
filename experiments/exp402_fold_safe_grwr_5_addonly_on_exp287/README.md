# exp402_fold_safe_grwr_5_addonly_on_exp287

## 状態

- Route: `ml_model`
- 状態: 分割Stage 0 technical gate PASS。Stage 1 GPU train version 4実行中
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`
- CV / Public LB / Private LB: 未実行
- steering:
  `.steering/20260726-exp402-fold-safe-grwr-5-addonly-on-exp287/`
- 正規train notebook: Stage 1 compact self-contained候補を採用
- 正規inference notebook / `settings.py`: 未編集placeholder
- 別名compact self-contained train/inference候補: 実装済み
- 専用test: 13件PASS
- Kaggle Stage 0: version 1 / id_no `128627922` /
  `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`
- Kaggle Stage 0 aggregate: version 2 / id_no `128831850` /
  `KernelWorkerStatus.COMPLETE`
- Kaggle Stage 1 train: version 4 / id_no `128627922` /
  `KernelWorkerStatus.RUNNING`
- preflight: `18 / 18` checks PASS、status `zero_booster_preflight_passed`
- 学習: 実行中。推論、提出: なし
- aggregate生成物:
  - partition manifest: 10 outer-role partitions
  - preflight manifest SHA:
    `c8af15ad8502b172031eaa862878ba07f2a94b6eba5913259c3a4ba0e5142de8`
  - reproducibility manifest SHA:
    `5456cfd9b0d2df3cac5848cb234cf382f9ebd1515439742fcff6afa3f0560fda`
- monolithic version 1終了原因: run/check時刻からruntime上限が最有力。ただしKaggle APIは
  手動cancelとruntime-limit cancelを区別しないため確定扱いしない

## 仮説

exp287でfold-safe化した`tvt_dense_d / tvt_densew_d / tvt_dense50_d`と、
既存のtarget-freeなPF/Beam 5候補から作る候補TVTの標準偏差・rangeは、
個別候補値だけでは表現しにくい局所的不確実性を表す。既存のGR/DWT/FFT成分との
固定interactionを含む5列をexp287へadd-onlyすれば、親の特徴面を変えずに
downstream TVT RMSEを改善できる可能性がある。

## 変更点

- exp287の421列は固定する。
- 旧exp218値を使わず、各outer foldのexp287 formation roleからGRWR 5列を再計算する。
- 追加列:
  - `grwr_candidate_tvt_std`
  - `grwr_candidate_tvt_range`
  - `grwr_dwt_energy_ratio_w065_x_candidate_std`
  - `grwr_fft_rotation_ratio_x_candidate_range`
  - `grwr_dwt_minus_raw_ncc_gap_x_candidate_range`
- 最終特徴数は`421 + 5 = 426`。
- exp396 entropy依存の6列目、score系27列、sample weight、hard gateは追加しない。

## 検証方針

- Fold: exp287と同じouter 5-fold GroupKFold
- Group: `well`
- Metric: score rowsの非加重RMSE
- Control: 保存済みexp287 OOF。親controlの再学習なし
- clean tail control: corrected exp264保存済みOOF
- Preflight: 0 model / 0 boosterでouter-role/current-test生成、schema、finite、SHA、
  formation reference境界、実行量をfail-closed検証
- 学習量: 1 variant × 3 LightGBM configs × 5 folds = 15 GPU boosters
- Promotion: pooled、4/5 folds、全scope、by-well p95、worst-well、
  exp264比悪化well数の固定AND gate
- Leakage check: historical GRWR値、full-train formation OOF、outer-valid formation、
  exp111/exp396 score、truth/error由来のfeature選択が0であること

## 実行入口

- 正規学習 notebook（Stage 1採用済み）:
  `exp402_fold_safe_grwr_5_addonly_on_exp287_train.ipynb`
- 正規推論 notebook（placeholder、未採用）:
  `exp402_fold_safe_grwr_5_addonly_on_exp287_inference.ipynb`
- Stage 0実装候補:
  `exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_train.py`
  と同名`.ipynb`
- Stage 1実装候補:
  `exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_stage1_train.py`
  と同名`.ipynb`
- fail-closed inference候補:
  `exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_inference.py`
  と同名`.ipynb`
- Stage 0とStage 1 trainのpackage/push/runは承認済み。inferenceとsubmissionは
  別のユーザー承認を必要とする。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |
| Stage 1 | version 4実行中、15 T4 boosters予定、control再学習0 |

## 所見

### 良かった点

- 5列の無効理由を、exp287のfold-safe formation生成物で解消できる設計に分離した。
- 既存421列とモデル設定を固定し、GRWR-5の純増分だけを比較できる。

### 未確認点

- Stage 0 technical gateはPASSし、GPU学習は実行中。GRWR-5のCV追加価値と
  promotion gateは完了後に確認する。
- 元となるdense 3列とGR/DWT/FFT成分が既に親surfaceにあり、冗長な可能性がある。

### リスク / 注意

- exp287はglobal改善に対してwell-tail guardを失敗しているため、global RMSEだけでは昇格しない。
- 旧exp218のGRWR block全体の改善から、この5列だけの効果を帰属できない。
- entropy interactionの6列目はexp396の閉鎖済みscore familyと混ざるため対象外。

## 次

Stage 1 version 2はaggregate inputのmount path解決で10.6秒後、0 boosterのまま
失敗した。version 3はmanifest名と固定file SHAでmount rootを選び、物理T4を
確認したが、clean-273再構築に必要なexp145 input不足で227.3秒後、0 boosterの
まま失敗した。version 4は`exp145-train`を追加し、固定11 inputと必要3ファイルを
前処理前に検証して再実行中。完了後にCVと固定promotion gateを確認し、PASS時だけ
inference実装を別途相談する。submissionも別承認境界を維持する。
