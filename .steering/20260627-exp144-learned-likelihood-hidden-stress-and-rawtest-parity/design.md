# 設計

## アプローチ

exp127 は shared rows 757,738 rows / 155 wells で learned likelihood add-only feature が control を改善した。ただし subset 評価で、hidden-like stress と raw-test/full-train parity が未確認だった。

exp144 では exp127 を再学習せず、保存済み row-level OOF predictions を読む。exp115 の hidden-like split artifact と exp112 の learned likelihood feature cache を join し、control (`exp092_shared_row_control`) と add-only (`learned_likelihood_confidence_addonly`) を同じ行集合で比較する。

## 実験範囲

- 対象実験: `exp144_learned_likelihood_hidden_stress_and_rawtest_parity`
- Route: `ml_model`
- 親実験: `exp127_learned_likelihood_features_on_exp092`
- 変更する変数: なし。readout と parity checklist のみ。
- 固定する変数: exp127 の予測、exp112 feature cache、exp115 split。

## 出力

- `overall_metrics.csv`: all shared rows と exp115 stress split の RMSE/MAE。
- `bucket_metrics.csv`: eval rank、md_since、spatial/eval/prefix/GR/TVT/typewell、learned likelihood confidence bucket 別。
- `by_well.csv`: well 別 score。
- `overall_delta.csv` / `bucket_delta.csv` / `worst_well_delta.csv`: add-only minus control。
- `rawtest_parity_checklist.csv`: full-train coverage、raw test feature regeneration、schema、submission candidate の pass/fail。
- `summary.json`: input SHA、focus delta、decision。

## 再現性設計

- seed policy: no new RNG。保存済み CSV の deterministic join と groupby のみ。
- stochastic 処理の有無: exp144 内ではなし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。upstream exp112/127 の生成物を読む。
- 並列処理と乱数の関係: なし。
- CPU/GPU runtime: CPU。LightGBM 学習なし。
- train cache / test feature regeneration の SHA 記録方針: 入力 gzip は raw SHA と decompressed SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: 新規 model / prediction / submission は作らない。upstream prediction SHA と readout CSV SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` で support files を再生成する。

## リスク

- リークリスク: true TVT は保存済み予測の scoring のみに使う。feature bucket は exp112 target-free cache と exp115 metadata に限定する。
- CV/LB 不一致リスク: stress readout は Public LB 代替ではない。submit 判断には使わない。
- ランタイム/メモリリスク: exp127 predictions は数百万行だが必要列のみ読む。Kaggle CPU で実行できる設計。
- 再現性リスク: upstream exp127 は GPU 学習済み生成物なので、exp144 は deterministic anchor ではない。
