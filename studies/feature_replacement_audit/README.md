# Feature replacement audit for exp148 / exp092 lineage

作成日: 2026-07-04

## 目的

exp148 で後から足した learned likelihood features と、exp092 以前に増えた feature surface を、相関だけでなくコード上の生成式から確認した。目的は「改善処理を新規特徴量として追加したが、本来は既存特徴量を置き換えるべきだった列」を洗い出すこと。

参照した主な証拠:

- `studies/exp148_feature_correlation/README.md`
- `/tmp/kaggle-output/exp148_feature_correlation_audit_v2/exp148_feature_correlation_audit_feature_readout.csv`
- `/tmp/kaggle-output/exp148_feature_correlation_audit_v2/exp148_feature_correlation_audit_top500_pairs.csv`
- `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/config.yaml`
- `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/learned_likelihood_fulltrain_addonly_on_exp092.py`
- `experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py`
- `experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py`
- `experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py`
- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/result.md`

## 結論

高信頼で「追加ではなく置換、または片側削除すべき」と判断できるのは、まず 17 列。

この 17 列は exp148 の相関監査で既に「まず落としやすい候補」とした集合と一致する。内訳は、定数 2、learned likelihood candidate TVT の既存 delta / disagreement 再出力 7、U-projection 内の符号重複 5、U-projection と base の完全重複 1、public replay base の完全または実質重複 2。

```text
sc_trust
ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt
ll_candidate_tvt_beam_mean_minus_last_known_tvt
ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt
ll_candidate_tvt_hyb_minus_last_known_tvt
ll_candidate_tvt_likpf_mean_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt
ll_candidate_tvt_sc_ens_minus_last_known_tvt
tda0
dense_bias
uproj_beam_mean_resid
uproj_beam_med_resid
uproj_diff_pf_ancc_minus_pf_z
uproj_likpf_mean_resid
uproj_pf_ancc_resid
uproj_pf_z_resid
```

次点で、exp092 以前の public replay surface にある `tvtFw_*` vs `tvtF50_*`、`bww_*` vs `bw50_*` は、同じ「prefix 末端を重視した formation bias」系の近似重複なので、置換候補として ablation すべき。exp148 相関監査の 30-column near-dupe 版では `tvtF50_*` と `bw50_*` を落とす扱いが最も自然だった。

## exp148 learned likelihood add-only で置換漏れだった列

exp145 の learned likelihood generator は `candidate_tvt_*` を候補値そのものとして出力する。`candidate_tvt_beam_mean` などは learned model の新しい予測値ではなく、元の候補 TVT 値である。

- `CandidateSpec` は `name` と `column` だけを持つ。
- `candidate_values` は `out[spec.column]` をそのまま積む。
- `candidate_tvt_*` はその candidate values をそのまま feature cache に出す。

該当コード:

- `experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py:32`
- `experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py:136`
- `experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py:379`
- `experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/learned_likelihood_rawtest_feature_generator_parity.py:441`

exp148 はこの `candidate_tvt_*` に対して `minus_last_known_tvt` と `minus_likpf_mean_tvt` を両方作る。ここで既存列と完全一致する列が発生している。

該当コード:

- `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/config.yaml:98`
- `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/config.yaml:141`
- `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/learned_likelihood_fulltrain_addonly_on_exp092.py:795`
- `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/learned_likelihood_fulltrain_addonly_on_exp092.py:817`

高信頼の置換関係:

| 既存または先行特徴量を残す | 後から追加された重複列を落とす | 相関 | 平均 importance |
| --- | --- | ---: | ---: |
| `beam_mean_d` | `ll_candidate_tvt_beam_mean_minus_last_known_tvt` | 0.999999996 | 1775.2 vs 1123.7 |
| `likpf_mean_d` | `ll_candidate_tvt_likpf_mean_minus_last_known_tvt` | 1.000000000 | 2142.1 vs 902.3 |
| `pf_ancc_delta` | `ll_candidate_tvt_pf_ancc_minus_last_known_tvt` | 1.000000000 | 1430.3 vs 630.2 |
| `sc_ens_d` | `ll_candidate_tvt_sc_ens_minus_last_known_tvt` | 1.000000000 | 138.8 vs 37.0 |
| `hyb_d` | `ll_candidate_tvt_hyb_minus_last_known_tvt` | 1.000000000 | 181.9 vs 52.4 |
| `uproj_diff_beam_mean_minus_likpf_mean` | `ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt` | 1.000000000 | 1221.7 vs 564.1 |
| `uproj_diff_pf_ancc_minus_likpf_mean` | `ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt` | 1.000000000 | 1740.7 vs 736.1 |
| なし | `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt` | 定数 0 | 0 相当 |

解釈:

- learned likelihood の probability / expected-error / rank / entropy / weighted TVT は learned model 固有の情報なので、置換候補ではなく保持側。
- ただし `candidate_tvt_*_minus_last_known_tvt` は、元候補の delta を再出力しただけ。これは add-only ではなく既存 delta と置換すべきだった。
- `candidate_tvt_*_minus_likpf_mean_tvt` のうち beam / pf は、既に exp092 の U-space disagreement に同じ差分があるため、learned likelihood 側で追加すべきではなかった。
- `ll_candidate_tvt_sc_ens_minus_likpf_mean_tvt` と `ll_candidate_tvt_hyb_minus_likpf_mean_tvt` は高相関だが、exp092 には sc/hyb vs likpf の完全対応列がない。これは immediate drop ではなく ablation 候補。

## exp092 U-projection で発生した生成上の重複

exp092 は exp085 で選ばれた U-projection correction/disagreement を fullrun した実験。config 上も add-only feature surface として設計されている。

該当箇所:

- `experiments/exp092_u_projection_correction_disagreement_fullrun/config.yaml:59`
- `experiments/exp092_u_projection_correction_disagreement_fullrun/config.yaml:81`
- `experiments/exp092_u_projection_correction_disagreement_fullrun/result.md:23`

U-projection source は、既存 base feature の `pf_ancc`、`pf_z`、`beam_mean_d`、`beam_med_d`、`likpf_mean_d` から作られる。

該当コード:

- `_source_tvt`: `experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py:372`
- U-space 変換: `experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py:444`

ここで `resid = source_u - poly` と `corr = poly - source_u` を両方出しているため、完全な符号反転重複がある。

該当コード:

- `experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py:505`
- `experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py:514`

落とす候補:

| 残す | 落とす | 相関 | 平均 importance |
| --- | --- | ---: | ---: |
| `uproj_pf_ancc_corr` | `uproj_pf_ancc_resid` | -1.000000000 | 1352.3 vs 1200.2 |
| `uproj_pf_z_corr` | `uproj_pf_z_resid` | -1.000000000 | 1296.4 vs 1060.3 |
| `uproj_beam_mean_corr` | `uproj_beam_mean_resid` | -1.000000000 | 1148.3 vs 1038.4 |
| `uproj_beam_med_corr` | `uproj_beam_med_resid` | -1.000000000 | 1021.3 vs 904.6 |
| `uproj_likpf_mean_corr` | `uproj_likpf_mean_resid` | -1.000000000 | 1811.4 vs 1676.4 |

また、`uproj_diff_pf_ancc_minus_pf_z` は U-space で差分を取ると `pf_ancc - pf_z` になり、base の `pf_vs_z` と完全一致する。

| 残す | 落とす | 相関 | 平均 importance |
| --- | --- | ---: | ---: |
| `pf_vs_z` | `uproj_diff_pf_ancc_minus_pf_z` | 1.000000000 | 1688.8 vs 743.5 |

注意:

- `abs_resid` と `resid_mad` は符号反転ではないため保持。
- `uproj_absdiff_pf_ancc_pf_z` は `abs(pf_vs_z)` なので、`pf_vs_z` と完全同一ではない。落とす場合は別 ablation。
- `beam_mean` と `beam_med` の U-projection は相関 0.9876 程度で高いが、完全置換とは言えない。

## exp092 以前の public replay surface 内の置換候補

exp063/072 の full public replay surface は、Ravaghi-style base features と Pixiux likelihood-PF replay features をまとめた 196-column surface。exp072 はこれを train cache として固定しただけで、新しい feature 意味は exp063 の replay code にある。

該当箇所:

- `experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/config.yaml:14`
- `experiments/exp072_exp063_full_replay_feature_cache/config.yaml:13`
- `experiments/exp072_exp063_full_replay_feature_cache/config.yaml:34`

### 完全または実質重複

| 残す | 落とす | 相関 | 理由 |
| --- | --- | ---: | --- |
| `gr_vs_tw_anc` | `tda0` | 0.999999971 | `ANCH_OFFS` に 0 があり、`tda0` は `last_tvt+0` の typewell GR residual。`gr_vs_tw_anc` と同じ式。 |
| `dense_rmse` | `dense_bias` | 0.999999999 | dense ANCC residual の RMSE と bias。式は同一ではないが、train surface では実質同じ well-level診断。 |
| なし | `sc_trust` | 定数 | `len(kn)/200` を 0.6 cap しており、今回 surface では single-value。 |

該当コード:

- `ANCH_OFFS`: `experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py:531`
- `gr_vs_tw_anc` / `tda0`: `experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py:627`
- `sc_trust`: `experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py:561`
- `dense_rmse` / `dense_bias`: `experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py:583`

### formation bias の near-duplicate

`seg_b_well()` は同じ prefix residual `bv = ktvt + z - formation` から、full median、early、mid、last50 median、exponential weighted average を作る。

該当コード:

- `experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py:460`
- `experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py:571`

このうち `bww_*` / `tvtFw_*` は weighted prefix、`bw50_*` / `tvtF50_*` は last50。どちらも「prefix 末端をより重視した formation bias」という同じ改善意図の別実装で、実測相関は 0.999990 から 0.999993。平均 importance は全 formation で weighted 側が同等以上だった。

置換候補として落とす列:

```text
bw50_ANCC
bw50_ASTNU
bw50_ASTNL
bw50_EGFDU
bw50_EGFDL
bw50_BUDA
tvtF50_ANCC
tvtF50_ASTNU
tvtF50_ASTNL
tvtF50_EGFDU
tvtF50_EGFDL
tvtF50_BUDA
```

保持候補:

```text
bww_ANCC
bww_ASTNU
bww_ASTNL
bww_EGFDU
bww_EGFDL
bww_BUDA
tvtFw_ANCC
tvtFw_ASTNU
tvtFw_ASTNL
tvtFw_EGFDU
tvtFw_EGFDL
tvtFw_BUDA
```

これは完全一致ではないため、17-column minimal prune には入れず、near-duplicate replacement ablation として扱う。

### 高相関だが置換とは断定しない列

以下は相関は高いが、生成式・意味が違うため「追加ではなく置換」とは断定しない。

| ペア | 相関 | 判断 |
| --- | ---: | --- |
| `md_since` / `dxy` | 0.999997950 | trajectory geometry が単調なので高相関。MD と XY 距離は意味が違う。 |
| `gr_nrg` / `grm21` | 0.999921773 | rolling GR energy と rolling mean。GR signal の別集約。 |
| `tvt_dense50_d` / `tvt_densew_d` | 0.999539184 | dense ANCC の last50 vs weighted。近いが両方 importance が高いので ablation 推奨。 |
| `form_rng_d` / `form_std_d` | 0.996395546 | formation spread の range と std。情報は近いが同一ではない。 |

## rank-slot 系は既に replacement-only が検証済み

exp098 は PF/Beam/likelihood-PF 候補を rank-slot structured features として exp073 に add-only した。exp098 は exp073 / exp077 より改善したが exp092 には届かなかった。

- `experiments/exp098_selector_rank_slot_features_on_exp073/result.md:15`
- `experiments/exp098_selector_rank_slot_features_on_exp073/result.md:37`

exp105 は compact 22-column rank-slot に削ったが悪化した。

- `experiments/exp105_compact_rank_slot_features_on_exp098/result.md:17`
- `experiments/exp105_compact_rank_slot_features_on_exp098/result.md:29`

exp108 は top3 関連だけに static prune したが、exp098 full より悪化した。

- `experiments/exp108_topn_related_feature_prune/result.md:21`
- `experiments/exp108_topn_related_feature_prune/result.md:42`

最後に exp147 が今回の問題意識に最も近く、exp092 generated columns 22 を drop し、rank-slot 25 列で置換する replacement-only を検証している。結果は rejected。

- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/config.yaml:21`
- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/config.yaml:123`
- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/result.md:17`
- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/result.md:33`
- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/result.md:65`

判断:

- rank-slot は特徴量として有用だが、exp092 U-projection correction/disagreement の置換には弱い。
- 今回の「追加ではなく置換すべきだった特徴量」には、rank-slot は現時点では採用しない。

## exp092 以前で調査したが置換漏れとはしないもの

exp038, exp040, exp041, exp042, exp043, exp057, exp058 は public PF/Beam/Ravaghi/NCC/GR diagnostics を add-only feature として評価している。ただし config 上、direct PF/Beam replacement、direct GR values、train-only formation columns、Ridge/meta-stack などを明示的に除外している。

代表例:

- `experiments/exp038_ravaghi_public_sel15_features_single_lgbm/config.yaml:12`
- `experiments/exp040_ravaghi_pf_ancc_pfz_feature_ablation/config.yaml:12`
- `experiments/exp041_ravaghi_beam_exact_feature_ablation/config.yaml:12`
- `experiments/exp043_ravaghi_feature_family_ablation_matrix/config.yaml:12`
- `experiments/exp057_xgb_catboost_pf_confidence_only_features/config.yaml:13`
- `experiments/exp058_lgbm_pf_confidence_only_features/config.yaml:14`

これらは「改善処理を本番予測の置換として使う」実験ではなく、diagnostic / add-only feature の評価として設計されていた。exp148/exp092 feature reduction の対象としては、最終的に exp063/072 の 196-column surface に入った重複群だけを扱えばよい。

## 推奨する次の ablation

1. `drop_exact_replacements_17`
   - 上記 17 列を落とす。
   - 目的は情報を失わず feature 数を 294 から 277 へ減らすこと。
   - これは最も低リスク。

2. `drop_exact_plus_formation_last50_29`
   - 17 列に `bw50_*` 6 列、`tvtF50_*` 6 列を追加で落とす。
   - feature 数は 265。
   - weighted prefix 側を残す判断。相関と importance は支持するが、完全一致ではないので CV 確認が必要。

3. `drop_exact_plus_prior_near_dupes_30`
   - 既存の exp148 相関レポートと同じ 30-column near-dupe 案。
   - 17 列に `bw50_*`、`tvtF50_*`、`dxy` を追加する。
   - `dxy` は置換漏れではなく geometry 高相関なので、今回の目的では 2 より優先度を下げる。

落とさない方がよい列:

- learned likelihood の probability / expected-error / rank / entropy / weighted TVT 系。
- U-projection の `abs_resid`、`resid_mad`、`absdiff`、source spread。
- rank-slot 系で exp092 を置換する案。exp147 で悪化済み。

## corr_prune_sanity_readout_on_exp148

作成日: 2026-07-04

`corr_prune_sanity_readout_on_exp148` は、この README の手動監査を再実行可能な no-training readout に寄せた安全装置。Kaggle GPU 学習、推論、提出は行わず、保存済みの exp148 correlation audit、exp148 train/inference schema、feature importance、exp145 train/rawtest schema を読み直すだけにしている。

実行コマンド:

```bash
.venv/bin/python studies/feature_replacement_audit/corr_prune_sanity_readout.py
```

出力先:

```text
studies/feature_replacement_audit/outputs/corr_prune_sanity_readout_on_exp148/
```

主な出力:

- `corr_prune_sanity_readout_on_exp148_drop_candidates.csv`: exact 17、formation last50 follow-up、learned-likelihood slim review、U-projection slim review を evidence level 付きで分離した候補表。
- `corr_prune_sanity_readout_on_exp148_config_fragment.yaml`: 後続 `exact_replacement_prune_on_exp148` へ移植するための `drop_columns` 断片。初回 active variant は `drop_exact_replacements_17` のみで、formation last50 12列は disabled follow-up。
- `corr_prune_sanity_readout_on_exp148_code_references.csv`: 候補の生成元コード参照。
- `corr_prune_sanity_readout_on_exp148_exp148_train_inference_schema_diff.csv`: exp148 train/inference schema diff。今回の実行では non-both 0。
- `corr_prune_sanity_readout_on_exp148_exp145_train_rawtest_schema_diff.csv`: exp145 train/rawtest schema diff。今回の実行では non-both 0。
- `corr_prune_sanity_readout_on_exp148_summary.json`: 入力、候補数、schema parity、出力ファイルの要約。

今回の固定結果:

- exact prune: 17 列。期待 feature 数は 294 -> 277。
- formation last50 follow-up: 12 列。exact 17 と合わせると 294 -> 265。
- learned-likelihood slim review: 4 列。初回 exact prune には混ぜない。
- U-projection slim review: 14 列。初回 exact prune には混ぜない。
- exp148 train/inference schema parity と exp145 train/rawtest schema parity はどちらも差分 0。

この readout は OOF 改善を主張しない。後続の `exact_replacement_prune_on_exp148` では、まず `corr_prune_sanity_readout_on_exp148_config_fragment.yaml` の `drop_exact_replacements_17` だけを使う。
