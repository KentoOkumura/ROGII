# exp207_heatmap_mdn_overlapping_window_path_stitch_probe

## 状態

- ルート: pf_beam
- 状態: kaggle_train_v2_complete_diagnostic_only
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-06
- 親実験: exp202_heatmap_mdn_candidate_generator_probe

## 仮説

exp202 の heatmap MDN topK は center-row 候補 union の oracle headroom を大きく増やしたが、local window 候補のままでは trajectory として使えるか分からない。各 window の topK local path を target-free に stitch し、full-well candidate として物理的に破綻しにくい path を作れるなら、exp204 系 selector 候補追加とは別に PF/Beam route の candidate generator として使える可能性がある。

## 変更点

- exp202 v2 の `heatmap_candidate_paths_top10.npz` と sample index を入力にする。
- well 内 row_center 順に topK local path を beam stitch する。
- stitch score は center score、rank、path smoothness、overlap disagreement、gap boundary continuity のみで計算する。
- exp099 の既存 PF/Beam candidate cache と join し、covered rows 上で oracle headroom、distance bucket、by-well、coverage を読む。
- 推論 notebook は no-submit guard のみ。

## 検証方針

- Fold: exp202 / exp099 の train-side artifact に準拠。
- Group: well。
- Stratification: なし。
- Leakage Check: true TVT、oracle、abs-error、within10、target-in-grid は stitch score に使わない。target は stitched path 固定後の readout のみ。
- 注意: 現行 exp202 v2 artifact は 14 validation samples / well の sparse local window output であり、dense overlapping full-well artifact ではない。overlap / coverage が不足する場合は結果として記録する。

## 実行入口

- 学習 notebook: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_train.ipynb`
- 推論 notebook: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp207_heatmap_mdn_overlapping_window_path_stitch_probe`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| covered existing+stitched top3 oracle RMSE | 4.418699605 |
| covered existing union oracle RMSE | 5.154353660 |
| row coverage vs exp099 cache | 0.352337441 |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 所見

### 良かった点

- Kaggle CPU train v2 が COMPLETE。GPU 再学習なしで exp202 path artifact の stitch 診断を実行できた。
- covered rows 上では existing + stitched top3 が oracle RMSE `5.154353660 -> 4.418699605` に改善し、by-well で worse は 0。
- sparse artifact の overlap / coverage 不足を readout に含めたため、full-well dense trajectory と誤読しにくい。

### 悪かった点

- 現行入力は sparse validation sample で、source overlap は 773 wells 中 3 wells / 39 pairs だけ。
- stitched only top3 は oracle RMSE `50.798377042` と粗く、単独候補や直接置換には弱い。

### リスク / 注意

- positive でも direct replacement / softmax average / PF weight replacement / submit には進めない。
- exp207 自体は deterministic だが、upstream exp202 は GPU-trained artifact なので deterministic submission anchor ではない。

## 次

- この backlog は exp207 で完了として閉じる。
- 続けるなら、exp202 model artifact から dense stride window path を再生成する `heatmap_mdn_dense_stride_window_path_regeneration_probe` に切る。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
