# 要件

## 依頼

固定 128 seed の exp404 互換 PF 軌跡を、known-prefix affine posterior と fold-safe AR(1) covariance を持つ block 尤度で再採点・再集約する実験を設計する。今回は実験ディレクトリ、steering、バックログを作り、実装・Kaggle 実行は行わない。

## 検証する問い

exp427 が target-free shift-rank で affine uncertainty と AR(1) covariance の追加識別力を示した場合、その同じ observation family は実際の PF seed 軌跡の選別にも有効で、exp404/exp417 の Gaussian seed evidence 集約を CV と tail の両方で上回るか。

## 制約

- Route は `pf_beam` とする。
- exp427 の technical AND gate と scientific AND gate の完全 PASS、および artifact SHA 固定を実装開始の必須先行条件とする。
- exp427 が FAIL または未確定の間は `waiting_exp427_gate` とし、コードを実装しない。
- exp404 x1.0、500 particles、固定 128 seed の PF trajectory variant は一つだけ再生する。
- 四つの factorial readout は同一 trajectory bank、同一 raw-finite support、同一 block、同一 affine posterior、同一 fold rho を使う。
- scientific candidate は `affine_ar1` 一つだけ。`affine_iid` と `identity_ar1` は診断、`identity_iid` は matched factorial control とする。
- temperature は `5.0`、evidence は block の proper predictive log-density の総和とする。mean log-density は監査用にだけ保存する。
- Huber、self-GR、datum reinjection、transition/particle update の変更を混ぜない。
- 後半 truth は score artifact と prediction freeze 後の評価にだけ使う。
- 実装、notebook 本体、Kaggle push、学習・推論は本設計の対象外とする。

## 受け入れ基準

- exp427 の先行 gate、fixed 2x2 尤度、seed evidence の集計尺度、実行量、比較対象、昇格 gate が文書と config で一致する。
- full run は 1 PF variant、773 well、98,944 seed-well trajectories、49,472,000 particle starts と明記される。
- 四つの readout は trajectory generation を共有し、候補ごとの PF 再実行を行わない。
- 保存済み親 control の独立再実行を含まない。
- exp427未確定中は本実験を未実装・未実行に保ち、terminal FAIL時は
  `closed_prerequisite_failed`としてno-rescueで閉じる。

## 最終状態

2026-07-29にexp427 version 2の完了結果を監査した。technical / scientific
AND gateがともにFAILしたため、受け入れ基準の実装開始条件は不成立である。
exp431は未実装・未実行のままterminal closeとする。

## 非目標

- exp427 Stage 0 の結果を先取りしない。
- affine prior、rho、AR order、block length、temperature を探索しない。
- exp427 FAIL 後に同一 OOF で救済 variant を作らない。
- 本設計を prediction CV、LB、提出 evidence とみなさない。
