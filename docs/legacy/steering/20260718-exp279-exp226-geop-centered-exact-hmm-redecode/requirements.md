# 要件

## 依頼

exp226 の保存済み group-safe OOF を absolute baseline として exact HMM を再デコードし、
GR matching が別 mode へ逸脱した後も前ステップの offset を引きずる現象を、毎行の弱い
復元力で抑えられるか検証する。バックログ作成から実験実装、静的検証までを行う。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp226 の最終 `tvt_pred` ではなく、GR 補正前の `tvt_geop` だけを HMM unary の中心に使う。
- exp209 の grid、41 rate states、transition、Gaussian GR emission、known-prefix calibration、missing-GR 処理を固定する。
- unary は `sigma=20 ft`、`lambda=0.50`、log-likelihood clip `600` の 1 variant に固定する。
- exp226、exp209、exp072 likelihood-PF、exp263 fixed formula は保存済み予測を比較に使い、再生成・再学習しない。
- unknown suffix の true TVT は path が確定するまで decoder に渡さない。
- LightGBM config / trained fold / booster は `0 / 0 / 0`。GPU、inference、submission は無効化する。
- sigma / lambda / grid / process-noise の探索、output blend、selector、PF 併用はこの実験に含めない。

## 受け入れ基準

- 3,783,989 rows / 773 wells を重複・欠損なしで生成し、全予測が finite である。
- exp226 OOF の decompressed SHA と group-safe fold 契約を hard guard する。
- `tvt_geop` が全評価行で exp209 固定 grid 内に入ることを hard guard する。
- candidate 生成完了後にだけ true TVT を結合し、overall / fold / distance / hidden-like / by-well / persistent-offset recovery を保存する。
- exp263 fixed formula 8.238331 を promotion baseline とし、OOF 0.02 以上改善、3/5 folds 以上改善、near / 1000+ / hidden-like の各悪化 0.02 以下、worst-well 悪化 +0.25 ft 以下を全通過した場合だけ inference 検討対象とする。
- 初回 notebook 実行は Kaggle CPU とし、ユーザー承認前には push しない。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
