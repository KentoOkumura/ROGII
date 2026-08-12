# 設計

## アプローチ

exp208 の dense stride local path artifact を再利用し、exp207/208 の target-free stitch beam を top5 full-well paths まで実行する。既存の stitched rows は監査用に残し、後続 selector 用には別途 `full_well_candidate_paths.csv.gz` を保存する。

contract table には target-free な列だけを入れる。`md_from_ps` は exp099 cache との row alignment と distance bucket 診断用に join する。true TVT や oracle error は path 固定後の評価 CSV にだけ使う。

## 実験範囲

- 対象実験: `exp210_heatmap_mdn_full_well_path_generation_probe`
- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`
- 入力親: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- 比較対象: `exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: full-well artifact schema、top5 stitch output、contract validation、schema/coverage/physicality readout
- 固定する変数: exp208 dense local path generation、exp207/208 stitch score weights、exp099 candidate union、no model training

## 再現性設計

- seed policy: 新規 stochastic 処理なし。入力 artifact order と deterministic stitch beam のみ。
- stochastic 処理の有無: exp210 内にはなし。upstream exp202 GPU training / exp208 dense generation / exp099 PF/Beam cache は stochastic component として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: exp210 は既存 PF/Beam cache と heatmap local paths の診断のみ。新規 PF/Beam sampling はしない。
- 並列処理と乱数の関係: stitch / contract formatting は single-process pandas/numpy 処理で乱数を使わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled、internet disabled。
- train cache / test feature regeneration の SHA 記録方針: exp208 dense path input、exp099 cache、full-well candidate path gzip の decompressed SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest と submission は対象外。full-well candidate path を prediction-like artifact として SHA 記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後、metadata と kernel source、bootstrap 内 config の整合を確認する。

## リスク

- リークリスク: exp099 target を評価前に読み込むため、contract table に true TVT/error 系が混入しないことを schema/required column で確認する。
- CV/LB 不一致リスク: train-side diagnostic であり、LB への直接判断はしない。
- ランタイム/メモリリスク: exp208 dense path npz と exp099 cache を CPU で読む。local topK 5/10、output top5 に限定する。
- 再現性リスク: upstream exp202/208 artifacts に依存するため deterministic submission anchor とは扱わない。
