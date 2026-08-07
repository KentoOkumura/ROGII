# exp148 feature correlation audit

## 目的

exp148 の 294 特徴量について、特徴量間の Pearson 相関を見て、特徴量削減候補を洗い出す。

## 実行

- Kaggle kernel: `kentookumura/exp148-feature-correlation-audit` v2
- 入力:
  - exp072 full replay train feature cache
  - exp145 full-train learned likelihood feature cache
  - exp148 train feature schema / feature importance
- 行数: 全 3,783,989 行から seed 148 で 600,000 行を uniform sample
- projection features は全行で構築してから sample した
- 出力取得先: `/tmp/kaggle-output/exp148_feature_correlation_audit_v2/`

## 要約

- 特徴量数: 294
  - base replay: 196
  - U-projection: 44
  - learned likelihood: 54
- `abs(corr) >= 0.90`: 1,786 pairs
- `abs(corr) >= 0.95`: 1,731 pairs
- `abs(corr) >= 0.98`: 1,498 pairs
- `abs(corr) >= 0.995`: 910 pairs
- 定数または単一値:
  - `sc_trust`
  - `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt`

## まず落としやすい候補

完全重複、符号反転、または既存特徴量の再表現で、かつ平均重要度が相手より低いもの。

```text
sc_trust
ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt
dense_bias
ll_candidate_tvt_beam_mean_minus_last_known_tvt
ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt
ll_candidate_tvt_hyb_minus_last_known_tvt
ll_candidate_tvt_likpf_mean_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt
ll_candidate_tvt_sc_ens_minus_last_known_tvt
tda0
uproj_beam_mean_resid
uproj_beam_med_resid
uproj_diff_pf_ancc_minus_pf_z
uproj_likpf_mean_resid
uproj_pf_ancc_resid
uproj_pf_z_resid
```

補足:
- `ll_candidate_tvt_*_minus_last_known_tvt` は対応する既存 delta 特徴と完全重複している。
- `ll_candidate_tvt_*_minus_likpf_mean_tvt` の一部は対応する U-projection difference と完全重複している。
- `uproj_*_corr` と `uproj_*_resid` は符号反転の完全重複。
- `tda0` は `gr_vs_tw_anc` と完全重複。
- `dense_bias` は `dense_rmse` とほぼ完全重複。
- `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt` は定数 0。

## 次に試す候補

`abs(corr) >= 0.99999` まで広げると、低重要度側の unique drop は 28 列になる。定数 2 列も含めると 30 列。上の候補に加えて、主に `bw50_*` と `tvtF50_*` が入る。

```text
bw50_ANCC
bw50_ASTNL
bw50_ASTNU
bw50_BUDA
bw50_EGFDL
bw50_EGFDU
dxy
tvtF50_ANCC
tvtF50_ASTNL
tvtF50_ASTNU
tvtF50_BUDA
tvtF50_EGFDL
tvtF50_EGFDU
```

ただし `dxy` は `md_since` とほぼ同じだが importance が高いので、削除は ablation で確認する。

## 注意点

0.98 や 0.995 の connected component をそのまま prune すると危険。特に `pf_z`、`last_known_tvt`、formation surface 系、`bw_*` 系は transitive に大きな成分へまとまるが、全ペアが同義ではない。ここは「候補 family」として扱い、再学習 ablation で確認する。

## 推奨 next experiment

新しい pruning ablation を作るなら、最初は次の 2 variant がよい。

1. `drop_exact_constants_and_dupes_17`: 上の「まず落としやすい候補」17 列を削除。
2. `drop_near_dupes_30`: 17 列に加えて `abs(corr) >= 0.99999` の低重要度側 13 列も削除。

exp148 anchor は強いので、いきなり 0.98 component 全体を削るより、17 列 / 30 列 prune の CV と LB-safe inference parity を先に見る。
