# 設計

## アプローチ

exp072 の固定候補 surface と exp073 / exp092 OOF predictions を join し、各 row の候補 TVT 周辺で GR score curve を作る。score curve は候補値そのものと `[-25, -20, -15, 0, 15, 20, 25] ft` shifted value を prefix TVT_input 上の近傍 index に写し、horizontal GR の local window similarity で採点する。

出力する主な特徴:

- `grbm_top1_top2_margin`
- `grbm_peak_count`
- `grbm_peak_spacing_ft`
- `grbm_gap_to_shift_15/20/25ft_*`
- `grbm_score_entropy`
- `grbm_bimodality_score`
- `grbm_ambiguity_score`
- `grbm_ambiguous_flag`
- `grbm_flat_score_flag`
- `grbm_mode_commit_proxy`
- `grbm_midpoint_proxy`
- `grbm_likpf_midpoint_blend`

評価は `grbm_ambiguous_flag` / `grbm_flat_score_flag` / ambiguity quantile / margin quantile / entropy quantile ごとの exp073, exp092, likPF, PF/Beam, diagnostic proxy error を読む。

## 実験範囲

- 対象実験: `exp133_gr_bimodal_match_ambiguity_detector`
- Route: `pf_beam`
- 親実験: `gr_bimodal_match_ambiguity_detector` backlog
- 入力: exp072 feature cache、exp073 train OOF、exp092 train OOF、raw train horizontal GR / TVT_input
- 変更する変数: GR score curve ambiguity detector と diagnostic proxy のみ
- 固定する変数: PF/Beam/likPF 候補、exp073 / exp092 OOF predictions、true TVT scoring rows

## 再現性設計

- seed policy: `no_new_rng_gr_ambiguity_diagnostic`
- stochastic 処理の有無: なし
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072 cache を読むだけで、本実験では再実行しない
- 並列処理と乱数の関係: 並列処理なし、乱数なし
- CPU/GPU runtime と deterministic flags: CPU only、GPU 不要
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を記録する
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は生成しない
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` で config と support file を package する

## リスク

- リークリスク: true TVT は error readout のみに使う。score curve、threshold、mode selection に true TVT を使わない。
- CV/LB 不一致リスク: CV や LB を直接改善する実験ではない。後続の add-only feature 実験で別途検証する。
- ランタイム/メモリリスク: 3.8M rows に対して row x candidates x shifts の score を well 単位で計算する。candidate-long は保存せず、wide feature と summary に絞る。
- 再現性リスク: upstream exp072/073/092 artifacts の version と content SHA を記録する。
