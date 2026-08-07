# exp431_affine_ar1_seed_evidence_reaggregation セッションノート

## 目的

exp427 が完全 PASS した場合に限り、その affine + AR(1) block likelihood で固定 128 seed PF 軌跡を再集約する後続実験の設計を先に固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `closed_prerequisite_failed`
- exp427: `stage_0_completed_gate_failed_closed`
- CV / LB: なし
- 実装・Kaggle push: 禁止、未実施

## 2026-07-28 設計記録

- exp427 technical/scientific AND gate の完全 PASS を必須先行条件にした。
- PF variants は 1、likelihood readout は fixed 2x2 の 4 個。
- seed weight は block mean ではなく proper log predictive density の全 block/run 合計から作る。
- full 実行量は 773 PF well-runs、98,944 seed-well trajectories、49,472,000 particle starts。
- LightGBM config / fold / booster は `0 / 0 / 0`。
- 親実験の独立 full rerun は 0。
- 実装コード、実行 notebook、Kaggle kernel は作成していない。

## 再現性メモ

- seed policy: immutable well id × seed index 0..127 の stable SHA-256
- stochastic components: PF particle transition、resampling
- likelihood artifact: PASS した exp427 の content SHA を必須入力にする
- score reduction: block/run/row の固定順
- trajectory: readout 前に凍結し、四 readout で共通 SHA
- deterministic anchor: prerequisite未確定、再実行 parity 未確認のため false

## 2026-07-29 先行 gate 監査と terminal close

Kaggle kernel
`kentookumura/exp427-affine-ar1-whitened-gr-readout-train` version 2の
`COMPLETE` logsを確認した。exp427の判断は
`stage_0_failed_close_without_rescue`だった。

- technical gate: FAIL
  - eligible block fraction `0.721073584 < 0.75`
  - eligible well fraction、finite coverage、row identity、candidate count、
    fold rho、outer-valid exclusion、dense/Woodbury parity、truth-lateはPASS
- scientific gate: FAIL
  - `affine_ar1` MRR `0.386090045`
  - matched identity-iid MRR `0.388002620`
  - saved exp280 MRR `0.388146378`
  - matched / saved比の改善foldはMRR各`2/5`、top3各`1/5`
- exp427 scientific contract SHA:
  `75241052d0bdeba3dcbad6548167bb1193f4375b1035e8de625591d4fdb24773`
- exp427 target-free bundle SHA:
  `3cae530e8c2629eea16468383ae06edc3e971d1ed77fb3a4d8d71d4043ba8a4d`

steeringで事前登録した分岐どおり、exp431は
`closed_prerequisite_failed`として未実装のまま閉じる。ユーザーの実装依頼は
別途実装承認としては十分だが、必須のtechnical/scientific完全PASSを置き換えない。
same-OOF rescue、support/gate緩和、PF replay、preflight、full run、推論、提出は
行っていない。

## 実行量

- exp431 PF well-runs: `0`
- seed-well trajectories: `0`
- particle starts: `0`
- likelihood readouts: `0`
- LightGBM config / trained fold / booster / GPU: `0 / 0 / 0 / 0`
- 親control再実行: `0`

## 次のアクション

なし。exp431を再開しない。exp427の失敗原因を説明する必要が独立に生じた場合だけ、
`KAGGLE_DIRECTION.md`の低優先度P4
`affine_ar1_rank_failure_attribution_readout`を別承認で検討する。

## 最終検証

- Kaggle status / logs:
  exp427 version 2 `COMPLETE`、decision
  `stage_0_failed_close_without_rescue`を確認
- `metrics.json` / train・inference notebook JSON / `config.yaml`: parse PASS
- `make validate-exp EXP=exp431_affine_ar1_seed_evidence_reaggregation`:
  strict PASS
- `make validate-template`: PASS
- `pytest -q tests/test_kaggle_notebooks.py tests/test_scaffold.py`:
  `11 passed`
- `make update-summary`:
  parentを`exp404_scale5_sigma_gr_likelihood_pf_ablation`、statusを
  `closed_prerequisite_failed`として更新
- `review_exp_docs.py exp431 --root .`:
  core evidence categories present
