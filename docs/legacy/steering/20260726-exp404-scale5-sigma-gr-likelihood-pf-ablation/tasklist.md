# タスクリスト

## Design-only完了

- [x] `exp404_scale5_sigma_gr_likelihood_pf_ablation`を採番した。
- [x] routeを`pf_beam`、scientific parentをexp400、kernel parentを
  exp072 deterministic v2へ固定した。
- [x] primary A/Bを`gs×1.0 + scale_5`対`gs×1.3 + scale_5`へ固定した。
- [x] x1.0/x1.3を同じstable per-well seedで両方再生成するpaired設計にした。
- [x] 500 particles / 128 seeds / temperature 5 / exp072 dynamicsを固定した。
- [x] `pf_mean`はexp072/exp400 parity専用で、科学primaryにしないと固定した。
- [x] 2 variants / 1,546 PF wells / 197,888 seed-well trajectories /
  98,944,000 particle starts / 5 folds / booster 0を固定した。
- [x] truth-late freeze、technical gate、scientific gate、fail-close、
  禁止事項を実行前に固定した。
- [x] backlog、steering、experiment scaffold、experiment summaryを
  design-only状態で登録する。

## Implementation-only完了

- [x] implementation-only承認を得る。
- [x] exp400 compact self-contained trainの章立てと記載量を比較する。
- [x] Jupytext percent形式のcompact self-contained train候補を別名で作る。
- [x] fail-closed compact self-contained inference候補を別名で作る。
- [x] x1.0/x1.3共通seed、scale5固定、mean parity、truth-late、
  execution count、gateの専用testを作る。
- [x] Jupytext round-trip、py_compile、Ruff F821、専用test、
  strict experiment validationを通す。

## 実行完了

- [x] 正規Notebook採用の別承認を得る。
- [x] Kaggle push前に2 variants / 1,546 PF wells /
  197,888 seed-well / 98,944,000 particle starts /
  model・booster・HMM・Beam 0を再確認する。
- [x] metadataとbootstrap内config、CPU、internet、kernel sources、
  seed contractを照合する。
- [x] Kaggle private CPU package / push / runの別承認を得る。
- [x] private CPU version 1（id_no `128628818`）を開始する。
- [x] version 1のprediction freeze後technical failureから、同一prediction bytesと
  scientific contractを保持したversion 4 recoveryを完了する。
- [x] output取得後にinput、prediction logical content、
  decompressed output、artifact manifest SHAを記録する。
- [x] technical gate PASS / scientific gate FAILを判定し、
  global scale5 x1.3を同じOOFで救済せず閉鎖する。
- [x] inference / submissionを未実行のまま維持する。

## 現在禁止

- temperature / multiplier / clip / particle / seed / scale grid
- scale 3/8/12生成、meanへのprimary差替え
- HMM / Beam / model / blend / selectorによる救済
- inference / submission
