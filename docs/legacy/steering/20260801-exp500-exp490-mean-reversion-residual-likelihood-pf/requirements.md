# 要件

## 依頼

exp490で有効性が確認されたexp226 geometry中心へのK16区間half-life平均回帰を、
exp486のresidual-state likelihood-PFへ移植する実験を設計確定する。

このターンの範囲は、`KAGGLE_DIRECTION.md`へのバックログ登録、steering文書、
実験scaffold、設定・記録文書の作成までとする。PF実装、正規Notebookの作り替え、
contract test、Kaggle package / push / run、full OOF、inference、submissionは行わない。

2026-08-02、ユーザーはStage 0 safety 3件FAILと終端閉鎖を確認したうえで、
「Stage1を実装・実行してください」と明示した。この依頼をStage 0 gateへの限定的な
user overrideとして扱い、同じ固定1 variantのfull OOFだけを実装・実行する。Stage 0を
PASSへ再分類せず、inference、submission、parameter / gate救済は承認範囲に含めない。

## 仮説

exp486 residual likelihood-PFのoffset / offset-rate遷移中心へexp490と同じ
K16区間half-life平均回帰を加えると、誤ったparticle basinのpersistent errorを減らせる。
ただし正しい長期offsetを消す可能性があるため、平均改善だけでなくwell-tail safetyを必須とする。

## 制約

- Routeは`pf_beam`とする。
- 科学的PF親は`exp486_exp226_geometry_residual_likelihood_pf`の
  `slow_residual_offset_state`とする。
- 平均回帰機構の親は`exp490_geometry_centered_mean_reverting_offset_hmm`とする。
- 変更する要素は、exp486 residual PFのoffset-rateとoffset遷移中心へexp490と同じ
  `rho_t`を掛けることだけとする。
- PFのparticle数、seed数、初期分布、process noise、roughening、resampling、GR emission、
  欠損GR処理、temperature-5 seed aggregation、出力dtypeはexp486から固定する。
- Huber emission、exact-HMM grid、posterior smoothing、exp490 predictionはPFへ混ぜない。
- half-lifeは対象行が属するexp226 K16区間のMD span 1区間分に固定し、探索しない。
- active scientific variantは1件とし、no-reversion control、exp404、exp486、exp490は
  保存済み予測・指標をload-onlyで参照する。control PFを再実行しない。
- Stage 0はexp411 fixed32とexp410 PF sentinel12の重複なしunion 44 wellsによる
  機構・安全性preflightであり、CVや昇格判定とは呼ばない。
- Stage 0実装・実行とStage 1 full OOF実装・実行は、それぞれ別のユーザー承認を必要とする。
  Stage 1は2026-08-02の明示overrideで承認済みとする。
- unknown suffixの正解TVT、error、fold、persistent/control role、episode、hidden-like roleは、
  candidate prediction、target-free diagnostics、contract SHAをfreezeする前に読まない。
- exp498の失敗を踏まえ、GR confidence、geometry disagreement、early offsetによる
  well/row gateや適応的half-lifeを追加しない。
- 再現性は`docs/06_reproducibility.md`に従い、stable per-well/per-seed RNG、固定順序、
  input / config / code / contract / prediction / diagnostic SHAを記録する。
- 実装時もLightGBM、学習fold、booster、HMM、Beam、GPUは使わない。

## 受け入れ基準

- PF粒子の状態、初期化、`rho_t`、rate / offsetの更新順序、出力TVTが一意である。
- exp486から変更する変数と固定する変数、exp490から移植しない要素が分離されている。
- Stage 0 fixed44とStage 1 full OOFの実行量、technical / mechanism / scientific gate、
  fail-closed条件が事前登録されている。
- Stage 1候補がexp404だけでなくexp226 finalを少なくとも`0.02 ft`上回ることを、
  direct PF昇格条件に含む。
- pooled RMSEだけでなくfold、GR欠損、1000+、hidden-like 2面、by-well p95 / worst、
  exp408 / exp410 persistent episodeを同時に判定する。
- `config.yaml`の`experiment.route`が`pf_beam`、scientific variantが1、
  LightGBM config / trained fold / booster / HMM / Beam / GPUがすべて0である。
- 設計確定時点がdesign-only、未実装、未実行だった履歴と、後続承認後のStage 0 Kaggle実行、
  fail-close判定、例外承認されたStage 1、inference / submissionが未承認・無効であることが、steering、
  `config.yaml`、`README.md`、`SESSION_NOTES.md`、`metrics.json`から分かる。
- gzip生成物はraw gzip SHAとdecompressed content SHAを分離し、後者を主証拠にする。
- deterministic anchorとは扱わず、将来のfull実行とraw-test regenerationの再実行一致まで
  anchor昇格を禁止する。

## 次

Stage 0はtechnical 13/13 PASSに対してmechanism safety 3件FAILであり、このnegative evidenceと
fail-close判定は維持する。2026-08-02のユーザー明示overrideに限り、固定条件のStage 1 full OOFを
4 CPU shard + truth-late strict mergeで実装・実行する。Stage 1結果にかかわらずinference、
submission、same-fixed44 / same-OOF rescueへ自動で進まない。
