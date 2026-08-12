# HMM / PF / exp226 well pattern readout 2026-07-12

既存 OOF / by-well artifact を well 単位で結合した diagnostic study。
新規学習、提出候補、anchor 更新ではない。

## Inputs

- HMM: exp223 `hmm_selfgr_boost_only_a070_c100`
- PF primary: exp072 `likPF_mean`
- pure PF: exp072 `pf_ancc`
- exp226: train OOF by-well metrics
- 大外し閾値: RMSE >= 30
- 当たり閾値: RMSE <= 10

## Key Counts

| category | wells |
| --- | ---: |
| `pf_bad_hmm_good` | 7 |
| `hmm_bad_pf_good` | 7 |
| `hmm_bad_exp226_good` | 9 |
| `likpf_bad_exp226_good` | 4 |
| `pf_ancc_bad_exp226_good` | 19 |
| `any_gr_bad_exp226_good` | 28 |
| `hmm_and_pf_bad_exp226_good` | 0 |
| `exp226_bad_any_gr_good` | 3 |
| `exp226_bad_hmm_good` | 2 |
| `exp226_bad_likpf_good` | 3 |
| `exp226_bad_hmm_and_pf_good` | 2 |

## Main Findings

- `hmm_and_pf_bad_exp226_good` は strict 条件では 0 本。
- `hmm_bad_exp226_good` は GR 欠損、低 self-GR valid rate、高 HMM std が目立つ。
- `likpf_bad_exp226_good` は GR 欠損ではなく PF/likPF branch offset が主因に見える。
- `exp226_bad_any_gr_good` は donor 距離が大きく、GR 欠損が少ない。z/geometry donor が外れ、GR 系が補正したケース。
- 直接 replacement / global fixed blend ではなく、confidence / selector feature として使うのが安全。

## Outputs

- `joined_well_summary.csv`: 773 well の結合表。
- `category_wells.csv`: 条件に該当した well の long table。
- `category_summary.csv`: 条件別の主要メトリクス集計。
- `feature_summary.csv`: 条件別 feature median と全体 percentile。
- `typewell_context.csv`: 該当 well が属する typewell group の文脈。
- `source_manifest.json`: 入力 artifact と実行条件。

## Source Docs

- `docs/surveys/hmm_pf_exp226_well_pattern_readout_20260712.md` に人間向けの解釈を記録。

Rows in `joined_well_summary.csv`: 773
