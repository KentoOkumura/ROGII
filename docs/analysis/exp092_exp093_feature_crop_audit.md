# exp092/exp093 特徴量 crop 監査

日付: 2026-06-27

## 対象

`exp092_u_projection_correction_disagreement_fullrun` と
`exp093_pf_candidate_coverage_then_ranker_audit` で使っている特徴量について、
well path の序盤にある急な下降区間を特徴量作成時に除外すべきかを確認した。

Assumption: ここでの crop は、学習・評価対象行を削除することではない。
`TVT_input` が最後に finite な anchor の近くにある target-free な行だけを使って、
一部の統計量を作り直すことを指す。境界は推論時にも再現できる必要があるため、
例えば `MD >= anchor_md - 1000`、`MD >= anchor_md - 2000`、または
`MD/X/Y/Z/GR/TVT_input` の欠損パターンだけから決まる deterministic な landing 判定を使う。

## 結論

exp092/exp093 では、PF/Beam 生成そのものは crop しない。
crop-window 版を追加すべきなのは、known prefix 全体を要約している特徴量、
または known prefix 全体の GR を matching library として使っている特徴量に限る。

優先して crop 版を作る対象:

- `slp_all`, `slp_b_d_all`, `slp_z`, `ktvt_range`, `ktvt_std`
- `pfx_rmse`, `cal_a`, `cal_b`
- formation / dense の prefix bias 系:
  `tvtF_*`, `bw_*`, `bw_early_*`, `bw_mid_*`, `frm_rmse_*`,
  `tvt_dense_d`, `dense_rmse`, `dense_bias`
- full-prefix SC/NCC 系:
  `sc8_d`, `sc15_d`, `sc25_d`, `sc_cons_d`, `sc_ens_d`,
  `sc*_sc`, `hyb_d`, `tdsc*`, `sc_vs_beam`

crop または置換しない対象:

- row-local な geometry / log 特徴量:
  `md_since`, `frac*`, `z`, `dx`, `dy`, `dz`, `dxy`,
  `dzdmd`, `dxdmd`, `dydmd`, `gr`, `gr_d*`, `grm*`, `grs*`,
  `glag*`, `glead*`, `gr_env`, `gr_nrg`
- typewell だけから決まる特徴量:
  `tw_range`, `tw_gr_mean`
- exp092 の直接 PF/Beam/likelihood-PF candidate 出力:
  `pf_ancc*`, `pf_z*`, `beam_*`, `likpf_mean_d`, `tdpf*`, `tdbc*`
- exp092 の U-projection / disagreement 追加特徴量:
  `uproj_*_corr`, `uproj_*_resid`, `uproj_*_abs_resid`,
  `uproj_*_resid_mad`, `uproj_diff_*`, `uproj_absdiff_*`,
  `uproj_source_u_std`, `uproj_source_u_range`, `uproj_corr_std`,
  `uproj_corr_range`

## 根拠

exp092 が使う exp072/exp073 public replay feature cache は 196 base features。
exp092 はここに U-projection correction / disagreement 44 特徴量を追加し、
合計 240 特徴量で学習している。追加された U-projection 側のコードは、
evaluation tail 上の `md_since` に対して well ごとの polynomial を fit しており、
序盤の known prefix / build section を直接は使っていない。

exp092 で序盤区間の影響を受けやすいのは、
`kn = hw[TVT_input.notna()]` を要約している helper 部分である。主な該当箇所:

- `slp_all` は `robust_slope(kmd, ktvt)` で known prefix 全体を使う。
  一方、`slp_50` はすでに prefix 末尾 50 行だけを使っている。
- `affine_cal(kgr, tw_at_k)` と `pfx_rmse` は known prefix 全体の GR を使う。
- `seg_b_well(ktvt, z_kn, form_col)` は full / early / mid / late / weighted の
  prefix bias を返す。`tvtF50_*` / `bw50_*` はすでに late segment だが、
  `tvtF_*` / `bw_*` は prefix 全体、`bw_early_*` は明示的に early segment を使う。
- Dense ANCC bias 診断の `tvt_dense_d`, `dense_rmse`, `dense_bias` は
  known prefix 全体を使う。一方、`tvt_dense50_d` はすでに late bias 寄り。
- `multi_scale_ncc(kgr, ktvt, hgr, ...)` は known prefix 全体の GR trace を
  search library として使っている。

exp093 は trained feature model ではなく candidate coverage audit である。
候補集合は固定 exp072 candidate と self-GR candidate で構成される。
self-GR builder はすでに `prefix_tail_rows=2048` で prefix 末尾側に制限しているため、
ここは hard global crop を入れるより、`prefix_tail_rows` を 512 / 1024 / 2048 で比較する方が自然。

## 推奨する実験形

exp092 系の add-only ablation として作る。
既存 240 特徴量はそのまま残し、crop-window 版の特徴量を追加する。
最初の実験では既存特徴量を置き換えない。

推奨 window:

- `tail1000`: `MD >= anchor_md - 1000` を満たす known-prefix rows
- `tail2000`: `MD >= anchor_md - 2000` を満たす known-prefix rows
- `last50`: 既存 `slp_50` / `tvt_dense50_d` と比較しやすい末尾 50 行

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

合格条件:

- exp092 の保存済み OOF / Public LB anchor と比較する。
  既存 control の再学習は、明示承認がない限り行わない。
- 通常 OOF で改善し、可能なら exp115 hidden-like stress readout でも悪化しないことを確認する。
- inference port 前に feature importance と worst-well delta を確認する。

