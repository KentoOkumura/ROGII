# exp148 特徴量 crop 監査

日付: 2026-06-30

## 対象

`exp148_learned_likelihood_fulltrain_addonly_on_exp092` で使っている特徴量について、
well path の序盤にある急な下降区間を特徴量作成時に除外すべきかを確認した。

Assumption: ここでの crop は、学習・評価対象行を削除することではない。
`TVT_input` が最後に finite な anchor の近くにある target-free な known-prefix 行だけを使って、
一部の統計量や matching score を作り直すことを指す。境界は current-test hidden rerun でも
再現できる必要があるため、例えば `MD >= anchor_md - 1000`、`MD >= anchor_md - 2000`、
または known prefix 末尾 `last50` のような deterministic な条件だけを使う。

## 結論

exp148 では、まず `exp092` 由来の full-prefix 統計と、exp145/111 由来の
multi-observation likelihood score を crop-window 版として add-only 追加するのが自然。
既存 294 特徴量は置換せず残し、最初の実験では learned probability / expected error を
crop 版に置き換えない。

優先して crop 版を作る対象:

- exp092 由来の known-prefix 全体統計:
  `slp_all`, `slp_b_d_all`, `slp_z`, `ktvt_range`, `ktvt_std`,
  `pfx_rmse`, `cal_a`, `cal_b`
- exp092 由来の formation / dense prefix bias 系:
  `tvtF_*`, `bw_*`, `bw_early_*`, `bw_mid_*`, `frm_rmse_*`,
  `tvt_dense_d`, `dense_rmse`, `dense_bias`
- exp092 由来の full-prefix SC/NCC 系:
  `sc8_d`, `sc15_d`, `sc25_d`, `sc_cons_d`, `sc_ens_d`,
  `sc*_sc`, `hyb_d`, `tdsc*`, `sc_vs_beam`
- exp148 learned likelihood 由来の multi-observation score 系:
  `multiobs_score_*`, `multiobs_mae_*`, `multiobs_ncc_*`,
  `multiobs_score_max`, `multiobs_score_mean`, `multiobs_score_gap`,
  `multiobs_top1_source_id`, `multiobs_top1_mae`, `multiobs_top1_ncc`

crop または置換しない対象:

- PF/Beam / likelihood-PF candidate 生成そのもの:
  `pf_ancc*`, `pf_z*`, `beam_*`, `likpf_mean_d`, `tdpf*`, `tdbc*`
- row-local な geometry / log 特徴量:
  `md_since`, `frac*`, `z`, `dx`, `dy`, `dz`, `dxy`,
  `dzdmd`, `dxdmd`, `dydmd`, `gr`, `gr_d*`, `grm*`, `grs*`,
  `glag*`, `glead*`, `gr_env`, `gr_nrg`
- exp092 の U-projection / disagreement 追加特徴量:
  `uproj_*_corr`, `uproj_*_resid`, `uproj_*_abs_resid`,
  `uproj_*_resid_mad`, `uproj_diff_*`, `uproj_absdiff_*`,
  `uproj_source_u_std`, `uproj_source_u_range`, `uproj_corr_std`,
  `uproj_corr_range`
- exp148 の learned probability / expected error そのもの:
  `ll_learned_prob_*`, `ll_learned_pred_abs_error_*`,
  `ll_learned_error_*`, `ll_learned_prob_weighted_tvt_*`,
  `ll_learned_error_weighted_tvt_*`
- candidate TVT 値とその差分:
  `ll_candidate_tvt_*_minus_last_known_tvt`,
  `ll_candidate_tvt_*_minus_likpf_mean_tvt`,
  `ll_candidate_tvt_std`, `ll_candidate_tvt_range`

## 根拠

exp148 は exp092 の feature surface を親にしている。構成は、exp072/073 public replay
feature cache 196 列、exp092 の U-projection correction / disagreement 44 列、
exp145 learned likelihood confidence 54 列の合計 294 特徴量である。

exp092 由来の crop 対象は `docs/analysis/exp092_exp093_feature_crop_audit.md` と同じ。
`slp_all`、`pfx_rmse`、`cal_a/b`、formation / dense prefix bias、full-prefix SC/NCC は
known prefix 全体を要約または matching library として使うため、序盤 build section の影響を受けやすい。
一方、U-projection 側は evaluation tail 上の candidate U-space を `md_since` で polynomial fit
しており、known prefix 全体を直接要約しないため crop 対象にしない。

exp148 で新しく増えた learned likelihood 54 特徴量は、exp145 の 51 列 `ml_features` を
`ll_` prefix 付きで取り込んだもの。exp145 は exp111 保存済み classifier / expected-error model を
target-free transform として使う。exp111 model feature order は 48 列で、その中に
`multiobs_score_*`、`multiobs_mae_*`、`multiobs_ncc_*`、candidate 間 disagreement、
candidate TVT と `last_known_tvt` の差が入る。

crop の影響を受けやすいのは multi-observation likelihood の生成部である。
`build_multi_observation_candidate_frame()` は well ごとに raw horizontal file を読み、
`TVT_input` が finite な prefix 全体から `prefix_tvt` を作る。その後、
candidate TVT に最も近い prefix TVT index を探し、周辺 GR vector と eval row 周辺 GR vector を
MAE / NCC で比較する。さらに `prefix_tvt` の `min` / `max` を使った out-of-range penalty も入る。
したがって序盤 build section に急降下や異なる GR regime がある場合、candidate の対応先や
range penalty が anchor 近傍の実態からずれる可能性がある。

ただし、exp111 の learned probability / expected error model は full-prefix multiobs 分布で学習された
保存済み fold0 model である。crop-window multiobs をそのまま exp111 model に入れて
`learned_prob_tail*` や `learned_error_tail*` を再生成すると、モデル入力分布の差し替えまで同時に起きる。
そのため最初の実験では、既存の `ll_learned_prob_*` / `ll_learned_error_*` は残したまま、
crop-window multiobs score と full-vs-crop 差分を LightGBM への add-only feature として渡す方が
原因を切り分けやすい。

## 推奨する実験形

`prefix_crop_window_features_on_exp148` として、exp148 系の add-only ablation を作る。
既存 294 特徴量はそのまま残し、crop-window 版の特徴量だけを追加する。
最初の実験では既存特徴量を置き換えない。

推奨 window:

- `tail1000`: `MD >= anchor_md - 1000` を満たす known-prefix rows
- `tail2000`: `MD >= anchor_md - 2000` を満たす known-prefix rows
- `last50`: known prefix 末尾 50 行

追加候補の特徴量 family:

- Prefix slope / TVT stats:
  `slp_md_tail1000`, `slp_md_tail2000`, `slp_z_tail*`,
  `ktvt_range_tail*`, `ktvt_std_tail*`
- Prefix GR / typewell calibration:
  `pfx_rmse_tail*`, `cal_a_tail*`, `cal_b_tail*`
- Formation / dense bias:
  `bw_tail1000_<formation>`, `bw_tail2000_<formation>`,
  `tvtF_tail1000_<formation>_d`, `dense_bias_tail*`, `dense_rmse_tail*`
- SC/NCC:
  `sc8_tail*_d`, `sc15_tail*_d`, `sc25_tail*_d`, `sc*_tail*_score`
- Multi-observation likelihood:
  `ll_multiobs_score_tail*_<candidate>`,
  `ll_multiobs_mae_tail*_<candidate>`,
  `ll_multiobs_ncc_tail*_<candidate>`,
  `ll_multiobs_score_max_tail*`,
  `ll_multiobs_score_mean_tail*`,
  `ll_multiobs_score_gap_tail*`,
  `ll_multiobs_top1_source_tail*`,
  `ll_multiobs_top1_changed_from_full_tail*`,
  `ll_multiobs_score_full_minus_tail*_<candidate>`,
  `ll_candidate_outside_prefix_tvt_range_tail*_<candidate>`

合格条件:

- 保存済み exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960 を主基準にする。
  既存 exp148 control の再学習は、明示承認がない限り行わない。
- 保存済み exp092 `lgb1` CV 9.322479896 / Public LB 8.350 は旧基準として参照する。
- 通常 OOF で exp148 から改善し、near-row、`1000_plus` bucket、worst-well regression が悪化しないこと。
- 可能なら exp115 hidden-like stress readout でも悪化しないこと。
- inference port 前に feature importance、raw-test/current-test schema parity、
  current-test learned likelihood generation の hidden-safe flow を確認する。

## 後続分岐

add-only crop-window multiobs が改善する場合だけ、同じ exp 系の後続分岐として
crop-window multiobs を exp111 saved model に入れ直した `learned_prob_tail*` /
`learned_error_tail*` を評価する。この分岐は saved model の入力分布変更を伴うため、
最初の add-only diagnostic とは分けて扱う。

add-only が悪化または特徴量重複で効かない場合は、replacement-only を別分岐として検討する。
対象は exp092 由来の full-prefix 統計と multiobs score に限定し、PF/Beam candidate 生成、
U-projection、learned probability / expected error、candidate TVT 値を同時に置き換えない。
