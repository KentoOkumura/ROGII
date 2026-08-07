# 要件

## 依頼

HMM 状態変数案として signed-curvature exact HMM の backlog、実験ディレクトリ、
steering を作り、実装前の設計を確定する。

## 制約

- Route: `pf_beam`
- `docs/06_reproducibility.md` に従う。
- 実装、Notebook置換、Kaggle実行、推論、提出は行わない。
- geometry / prefixによる現在rate予測を前提にしない。
- exp209の曲率以外の状態・観測設定を固定する。

## 受け入れ基準

- c state、drift、遷移、Stage 0/1 gate、resource上限が一意である。
- 3軌道とGR scoreをtruth join前にfreezeする。
- 1 variant / 773 HMM runs / 0 booster / control rerun 0を記録する。
- backlog と summary に設計確定・未実装として登録する。
- 将来のgzip生成物はdecompressed content SHAを主証拠にする。

## 2026-07-25 実装承認

- ユーザーの `exp364を実装してください` を Stage 0 実装承認として扱う。
- 承認範囲は3本の固定signed path readout、truth-late-join、16-well resource projection、
  fail-closed inference候補、静的検証まで。
- Kaggle package/push/run、Stage 1 exact HMM、inference、submissionは承認範囲外。
