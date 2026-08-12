# 要件

## 依頼

`linear_md_z_prior_global_search` バックログを train-side diagnostic として実装する。最後の既知 TVT 点を起点に、`TVT = last_tvt + a*dMD + b*dZ` の弱い線形 prior が pseudo-tail true TVT を fold-safe に説明できるか確認する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- Kaggle Notebook 実行を正とする。ローカル notebook 実行はしない。
- `a,b` の fold 選択に validation well の true TVT を使わない。
- GR 照合 loss は補助診断に限定し、採用判断は TVT RMSE、distance bucket、worst-well、fold 外 selection で行う。
- exp065 native typewell group は一貫性診断に使い、valid true TVT を group prior fit へ混ぜない。
- target 変更や LightGBM 再学習、inference port、submission は今回の初期実装範囲外にする。

## 受け入れ基準

- `experiments/exp113_linear_md_z_prior_global_search/` に config、補助 `.py`、train/inference notebook、README、SESSION_NOTES、result、metrics がある。
- train notebook は入力確認、fold-safe grid search、metrics / 生成物保存をセル単位で追える。
- 生成物として candidate metrics、fold selection、bucket metrics、by-well metrics、native group consistency、OOF prediction/features、summary JSON を保存する。
- deterministic anchor として扱わないことを config と記録に明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
