# 設計

## アプローチ

exp418 Stage 0 summaryを新しい数値契約の入力証拠として読み、exp418をPASSへ
再分類せずに後継branchのeligibilityを判定する。summary file SHA、decision、
technical check集合、scientific threshold check集合、観測差を照合する。

学習前に、truth/error/oracleを使わないfixed synthetic 16-rate vectorsをmatrix basisと
canonical sequential accumulationの両方で積分し、最大差が`1.0e-10 ft`以下で
あることを確認する。その後だけexp418と同じ5-fold LightGBM Stage 1を実行する。

## 実験範囲

- 対象実験: `exp421_signed_segment_rate_numerical_contract_stage1`
- Route: `ensemble`
- 親実験: `exp418_exp226_signed_segment_rate_residual`
- 変更する変数: Stage 1 eligibilityのintegration parity上限
  `1.0e-12 → 1.0e-10 ft`、およびtruth-free synthetic numerical audit追加
- 固定する変数: target sign/unit、zero-intercept K16 cumulative least squares、
  destination-row segment assignment、first-row correction 0、continuous
  integration、exp333 nested fold / 136-feature surface、LightGBM `lgb1`、
  row-count sample weight、全Stage 1 scientific gate、seed、CPU runtime

## 再現性設計

- seed policy: exp418と同じfold identity、LightGBM `random_state=0`
- stochastic 処理の有無: CPU LightGBM学習のみ
- PF/Beam / likelihood-PF / seed bagging の有無: なし
- 並列処理と乱数の関係: fixed fold order、canonical well/row/segment sort、
  LightGBM `deterministic=true` / `force_col_wise=true`
- CPU/GPU runtime と deterministic flags: Kaggle private CPU、GPU off、
  `n_jobs=num_threads=8`
- train cache / test feature regeneration のSHA記録方針: exp333 nested、
  exp226 OOF、exp072 cache、feature schema/content/freeze SHAを記録。testは範囲外
- model manifest / prediction / submission SHA記録方針: 5 model file SHA、
  segment prediction content SHA、OOF content SHAを記録。submissionは生成しない
- Kaggle package bootstrap確認方針: metadataとembedded configのkernel source、
  selected stage、数値上限、5 booster、CPU/internet-offをpush前に照合

## リスク

- リークリスク: exp418 summaryはtruth join後だがeligibilityにだけ使い、特徴量や
  sample選択には使わない。特徴量はexp333と同じtruth-free freeze後にtruth joinする
- CV/LB不一致リスク: signed-rate誤差も長いsuffixで累積する。pooledだけでなく
  near/tail/hidden/boundary/by-well gateを維持する
- ランタイム/メモリリスク: 3.78M row feature surfaceと5 CPU boosters。
  exp226やcontrolを再学習しない
- 再現性リスク: CPU LightGBMは固定flagを使うがcurrent-test rerun parityがなく、
  deterministic submission anchorとは呼ばない
- ガバナンスリスク: exp418をPASSに書き換えない。exp421の`1e-10`契約でも
  eligibilityまたはStage 1 gateがFAILした場合はsame-OOF rescueせず終了する
