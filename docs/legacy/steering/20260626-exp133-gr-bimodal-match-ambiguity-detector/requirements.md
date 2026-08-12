# 要件

## 依頼

`gr_bimodal_match_ambiguity_detector` バックログを実装する。GR matching の +/-15-25ft decoy や複数 peak を target-free に検出し、exp073 / exp092 / likPF / PF/Beam 候補の error と突き合わせる train-side diagnostic を作る。

## 制約

- Route: `pf_beam`
- LightGBM や candidate selector の学習は行わない。
- exp072 の PF/Beam/likPF feature cache、exp073 / exp092 の train-side OOF predictions は入力として固定する。
- detector は真値側 mode を当てに行かない。mode commit と midpoint は診断 proxy としてだけ評価する。
- Public LB に合わせて decoy spacing や threshold を tuning しない。
- flat score well と genuinely bimodal well を分ける。
- 再現性: `docs/06_reproducibility.md` に従い、upstream gzip input の decompressed content SHA を記録する。

## 受け入れ基準

- `experiments/exp133_gr_bimodal_match_ambiguity_detector/` に config、train/inference notebook、補助 script がある。
- train notebook は input contract、leakage policy、実行結果、生成物 preview、metrics 保存をセルで追える。
- 生成物として row context、candidate metrics、bucket metrics、well metrics、GR ambiguity feature cache、summary JSON を保存する。
- `metrics.json` に rows、wells、ambiguous rate、flat rate、best candidate、input SHA、feature cache SHA を保存する。
- 実験は train-side diagnostic として記録し、推論・提出は無効化する。
