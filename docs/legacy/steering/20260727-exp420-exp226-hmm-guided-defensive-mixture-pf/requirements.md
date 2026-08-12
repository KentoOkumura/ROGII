# 要件

## 依頼

HMM、likelihood-PF、exp226 の誤差原因を踏まえ、3者の長所を単一の物理
アルゴリズムへ統合する次実験を設計する。

- exp226 は GR 補正前の fold-safe geometry 局所 rate だけを使う。
- HMM は untreated forward filter の predictive-to-filtered rate innovation から
  作る方向付き schedule だけを使う。
- PF は元 transition を 50% 残す importance-corrected defensive mixture、
  500 particles、128 stable seeds、temperature-5 evidence aggregation を使う。
- 最終予測は統合 PF の単一出力とし、HMM / exp226 prediction の blend、
  selector、学習済み ML model は使わない。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- HMM posterior mean、backward message、absolute TVT path は PF state / outputへ
  入れない。
- exp226 final prediction、GR correction、U projection、absolute path は使わない。
- PF target transition、position conditional、GR emission、resampling、roughening、
  particle / seed 数は exp404 x1.0 scale5 系から変更しない。
- rate proposal は元 transition を常に 50% 含み、`p0/q <= 2` を構成上保証する。
- Stage 0 は固定済み 32 HMM wells と 12 PF sentinel wells の重複なし union
  44 wells とし、同じ OOF を見た threshold / duration / proposal weight /
  width の救済探索を行わない。
- 保存済み exp404、exp226、exp209 control は再実行しない。
- 実装、正規 Notebook 編集、Kaggle package / push / run、inference、
  submission は今回の承認範囲外とする。

## 2026-07-27 実装承認

後続のユーザー指示`exp420を実装してください`により、train-sideのcompact
self-contained候補、contract tests、fixed44 / full orchestrationの実装が承認された。
正規Notebook採用、Kaggle package / push / run、inference、submissionは引き続き
承認範囲外とする。

## 受け入れ基準

- `docs/legacy/steering/20260727-exp420-exp226-hmm-guided-defensive-mixture-pf/`、
  `experiments/exp420_exp226_hmm_guided_defensive_mixture_pf/`、
  `KAGGLE_DIRECTION.md`、`experiment_summary.md` に同じ設計境界が記録される。
- `config.yaml` の `experiment.route` が `pf_beam` であり、active scientific
  variant が1、LightGBM config / trained fold / booster / GPU がすべて0である。
- untreated HMM schedule、inactive / active proposal、importance correction、
  temperature-5 seed aggregationの式と固定値が一意に定義される。
- Stage 0、full OOF、mechanism / standalone / physical-anchor gate、
  fail-close decision が事前固定される。
- proposal生成前に読める列と、candidate freeze後だけ読める truth / error /
  episode / scope列が分離される。
- deterministic anchor として扱う場合は、input / code / config / schedule /
  prediction / diagnostic のSHAとKaggle kernel version、fixed-probe rerun parityが
  記録される。model / submissionを生成しない段階ではmodel / submission SHAは
  非該当と明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
