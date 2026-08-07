# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先度 backlog `pf_seed_medoid_selectability_audit` を
`exp252_pf_seed_medoid_selectability_audit` として実装する。

exp243 v3 exact-parity run が保存した base8 + K8 medoid row candidates、K8 cluster
manifest / summary、PF diagnosticsだけを固定入力とし、K8が追加したtrajectory-mode
headroomをtarget-free診断で識別できるか監査する。

## 制約

- Route: `pf_beam`。
- no-training diagnostic。LightGBM config 0、fold 0、booster 0、PF replay 0。
- Kは8だけを使用し、K3/K5を再追加しない。
- base8とK8の候補値、cluster assignment、medoid、likelihood、PF診断を変更しない。
- true TVTはtarget-free scoreを固定した後のlabel、AUC、regret、coverage評価だけに使う。
- score / threshold grid、selector学習、候補平均、candidate追加、raw-test PF再生成、
  inference、submissionを行わない。
- shuffled-score negative controlを全scope・全scoreに対して実行する。
- `docs/06_reproducibility.md` に従い、exp243入力のdecompressed content SHAと
  stable shuffle seedを記録する。

## 受け入れ基準

- exp243 canonical v3のrow candidates、cluster manifest、cluster summary、PF diagnostics
  を期待SHA付きでpreflightする。
- bank-selectabilityとcandidate-selectabilityを分離し、target-free score contractを
  configとnotebook上に表示する。
- row、128/256/512 contiguous block、whole-wellについて、best-source label、AUC、
  top1 regret、coverageを保存する。
- bank scoreにはcluster entropy/HHI、assignment distance、ESS/resampling、likelihood
  dispersion、seed spread、base8 disagreementを含める。
- candidate scoreにはseed/likelihood mass、likelihood rank/gap、within/separation distance、
  base path disagreementを含める。
- real scoreとstable shuffled-scoreを同じ評価実装で比較し、target-conditionedなscore選択を
  行わない。
- inference notebookは明示的に停止し、submissionを生成しない。
- Jupytext percent source、ipynb変換、py_compile、Ruff、strict experiment validation、
  Kaggle package bootstrap/config一致を確認する。
- gzip入力はraw gzip SHAではなくdecompressed content SHAを主証拠とする。
