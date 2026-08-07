# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `typewell_late_range_ml_posthoc_clip_audit` を実装する。

## 制約

- Route: `ml_model`
- 学習なしの train-side OOF posthoc audit に限定する。
- 主対象は `exp148_learned_likelihood_fulltrain_addonly_on_exp092` の OOF prediction。exp092 / exp073 は入力 prediction が利用できる場合だけ同じ監査関数で追加評価できる optional source とする。
- `known_last_pct` が高い well で、ML 予測の `pred_pct` が typewell TVT range 前半へ落ちる行だけを条件付き shrink / clip する。
- `pred_pct >= known_last_pct` のような単調制約は置かない。
- lower bound は fixed pct と `known_last_pct - margin` の小 grid に限定する。
- true TVT は scoring と bucket readout にだけ使い、gate 条件や lower bound 算出には使わない。
- inference port / submit は実装しない。positive でも raw-test parity と worst-well guard の追加確認が必要。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp174_typewell_late_range_ml_posthoc_clip_audit/` に config、監査 module、train notebook、記録ファイルがある。
- `exp148` prediction source から `pred_pct`、`target_pct`、`known_last_pct`、`lower_bound_pct`、変更量を materialize できる。
- baseline と clip / shrink grid の overall metrics、bucket metrics、by-well metrics、group metrics、changed-row summary、source coverage summary を出力する。
- near `000_050`、`1000_plus`、test-like subset、front-half exception stress、worst-well regression を確認できる。
- Kaggle push 前の計算規模として、variant 数、LightGBM config 数 0、fold 数 0、booster 数 0、control 再学習なしが `SESSION_NOTES.md` と config に記録されている。
- deterministic anchor として扱わない理由が config / result / summary に記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
