# exp398_all_well_1p3_sigma_gr_exact_hmm

## 状態

- ルート: pf_beam
- 状態: train_side_all_well_sigma_x1p3_gate_failed_closed
- CV: 12.710664241676811
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-25
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp209 HMMのGR evidenceが全体に強すぎるなら、selectorを使わず全wellの
`sigma_gr`を`1.3`倍するだけでunknown-suffix RMSEを改善できる。

## 変更点

- exp209 known-prefix zero-fill population stdを`[10,60]`でclipする。
- clip後のscaleへ全well共通`1.3`を1回だけ掛け、再clipしない。
- 有効scale範囲は`[13,78]`。
- Gaussian emission、state/rate grid、transition、prior、missing-GR、Type Well補間、
  posterior meanはexp209から変えない。
- saved exp209 HMMとLikPFはcontrolとして読み、再実行しない。

## 検証方針

- Fold: exp226保存済み5 reporting folds
- Group: well_id
- Metric: unknown-suffix row RMSE
- Leakage Check: candidate predictionとcontent SHAをtruth/control join前にfreeze
- Guard: overall、fold、raw observed/missing、high-missing、1000+、hidden-like 2面、
  by-well p95/worst、fixed LikPF 50:50

## 実行入口

- 学習 notebook: `exp398_all_well_1p3_sigma_gr_exact_hmm_train.ipynb`
- 推論 notebook: `exp398_all_well_1p3_sigma_gr_exact_hmm_inference.ipynb`
- train候補:
  `exp398_all_well_1p3_sigma_gr_exact_hmm_compact_selfcontained_train.py`
- inference候補:
  `exp398_all_well_1p3_sigma_gr_exact_hmm_compact_selfcontained_inference.py`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp398_all_well_1p3_sigma_gr_exact_hmm`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | `12.710664` |
| saved exp209 control | `11.938287` |
| 改善量 | `-0.772377 ft` |
| 改善fold | `1 / 5` |
| fixed LikPF 50:50 | `10.653104`（control `10.269693`、`-0.383411 ft`） |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- exp397の不安定なprefix selectorを持ち込まず、外部主張に近い全well固定`1.3`を
  単一変更として直接検証できる。
- saved exp209 controlを再実行しない。
- 3,783,989 rows / 773 wellsを約5時間22分で完走し、truth-before-freeze 0。

### 悪かった点

- overallは`0.772377 ft`悪化し、改善は1/5 folds、330/773 wellsだけだった。
- required scope、by-well p95/worst、fixed LikPF 50:50をすべてFAILした。

### リスク / 注意

- 実行済み倍率監査はCSV round-trip差`2.13e-14`を`atol=0`で比較して偽陰性になった。
  全773記録の倍率は`1.3`で、ローカル監査だけ`atol=1e-12`へ修正した。
- 科学結果は監査偽陰性と独立して大幅悪化しており、再実行しない。

## 次

- Kaggle private CPU version 1（id_no `128542706`）完了。
- `all_well_sigma_x1p3_failed_close_without_rescue`としてbranchを閉じる。
- inference、submission、version 2、parameter rescueは行わない。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
