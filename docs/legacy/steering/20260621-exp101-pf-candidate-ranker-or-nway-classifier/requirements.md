# Requirements

`pf_candidate_ranker_or_nway_classifier` を実装する。exp099 v2 の multi-observation likelihood wide cache を固定入力にし、PF/Beam/likelihood-PF 5候補から supervised ranker / N-way classifier で候補 index を選ぶ train-side audit を行う。

## Context

- Route: `pf_beam`
- Parent: `exp099_pf_multi_observation_likelihood_probe`
- Cache parent: `exp099_pf_multi_observation_likelihood_probe`
- 再現性: `docs/06_reproducibility.md` に従い、入力 gzip の decompressed SHA、schema SHA、model manifest、OOF prediction SHA を保存する。

## Scope

- 候補は最初は `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb` に限定する。
- 教師 label は train pseudo-tail true TVT から作る oracle-best candidate index とする。
- feature は target-free に限定し、`target`, `true_tvt`, `oracle_label`, `oracle_candidate` を入れない。
- exp099 の `multiobs_score_*`, `multiobs_mae_*`, `multiobs_ncc_*`, score gap/top1 source id を特徴量として使う。
- `multiobs_top1` / softmax / blend 系は direct prediction として採用しない。
- 比較対象は `likpf_mean` 単体、target-free `multiobs_score_top1`、oracle、LightGBM multiclass、candidate-long binary scorer、candidate-long predicted-error ranker。

## Acceptance Criteria

- `experiments/exp101_pf_candidate_ranker_or_nway_classifier/` に設定、補助コード、train notebook、inference notebook、記録ファイルがある。
- train notebook は設定、入力確認、候補/特徴量確認、実行、metrics/生成物 preview をセル単位で追える。
- 実行時に次の生成物を保存できる。
  - metrics CSV
  - OOF selected predictions CSV.GZ
  - selection distribution CSV
  - by-well path switch CSV
  - bucket metrics CSV
  - feature importance / mean importance CSV
  - feature schema CSV
  - model manifest JSON
  - summary JSON
- summary JSON に input cache SHA、decompressed SHA、schema SHA、prediction SHA、model SHA、decision が含まれる。
- この実験は提出しない。良い結果が出ても continuity / worst-well / raw-test feature parity の follow-up を必要条件にする。

## Non-goals

- TVT 直接回帰を学習しない。
- 候補値の soft average を採用しない。
- hidden branch 置換や PF/Beam 再生成はしない。
- inference port / submission はしない。
