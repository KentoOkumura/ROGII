# 設計

## アプローチ

raw train horizontal wells で、各 well の consecutive step から `dTVT ~= a * dZ + b` を最小二乗 fit する。`b` は formation level / offset を表す target と見なし、train full true TVT 由来の `b` peak distribution を診断する。v1 の `dTVT/dMD ~= a*dZ/dMD+b` rate-fit は Public LB 41.214 で失敗したため、v2 では discussion 本文の step increment へ修正した。

train pseudo-tail では、validation target well を source pool から除外する。query 側は known prefix の last-300 rows から prefix `a,b` と TVT/Z slope/delta を作る。source 側は target 以外の train full-fit `a,b` を使い、same peak / nearest XY / exact typewell hash に加えて、X/Y/Z + last-300 TVT/Z feature-nearest と visible prefix holdout selector で local `a,b` または fixed-a source `b` を選ぶ。未知 suffix は last known TVT から `dTVT = a*dZ + b` を累積して予測する。

test inference では全 train full-fit `a,b` を source pool とし、test well は prefix fit、last-300 TVT/Z summary、typewell exact hash、X/Y/Z geometry のみで同じ assignment を行う。v2 selected candidate は train-side best の `prefix_holdout_source_b_fixeda_h600` とし、直接 `submission.csv` を作る。

## 実験範囲

- 対象実験: `exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline`
- Route: `pf_beam`
- 親実験: `KAGGLE_DIRECTION.md` backlog `discussion711308_dz_dtvt_bpeak_cluster_baseline`
- 変更する変数: fit equation、`a,b` source selection rule、`b` peak assignment、nearest XY fallback、exact typewell hash fallback、feature-nearest source selection、prefix-holdout selector。
- 固定する変数: raw data schema、score rows (`TVT_input` missing suffix)、last known TVT anchor、no ML training、no PF/Beam rerun、no blend with current ML anchors。

## 再現性設計

- seed policy: no RNG。peak center detection、nearest-neighbor、aggregation は deterministic sort で tie break する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。route は PF/Beam baseline 管理だが、実装は no-ML deterministic rule。
- 並列処理と乱数の関係: 並列処理なし。global RNG 不使用。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled、internet disabled。
- train cache / test feature regeneration の SHA 記録方針: train OOF prediction gzip は raw gzip SHA と decompressed content SHA を記録し、decompressed を主証拠にする。submission は CSV content SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model artifact はなし。train OOF prediction SHA、submission SHA、assignment summary SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に metadata と embedded support files を確認する。

## リスク

- リークリスク: full train `b` peak label を target validation well の割当に使うと oracle になる。実装では target well を source pool から除外し、query は prefix fit / last-300 summary のみで peak label と source selection を決める。
- CV/LB 不一致リスク: train source full-fit `a,b` は true TVT を持つ train wells だけから得るため、test では neighbor / cluster 割当品質がずれる可能性がある。LB 約 12.8 の再現が目的で、ML anchor と直接混ぜない。
- ランタイム/メモリリスク: 3.8M train unknown rows に複数 candidate prediction を保存するため memory は数百 MB 程度。LightGBM や GPU は使わない。
- 再現性リスク: gzip metadata は raw SHA が変わり得るため decompressed SHA を主証拠にする。

## v2 結果

- train v2 best: `prefix_holdout_source_b_fixeda_h600` RMSE 35.41055512960111
- inference v2 / submit ref: `54396544`
- Public LB: 34.908
- 結論: v1 41.214 からは改善したが、要件 LB 約 12.8 を大きく未達。standalone direct baseline としては採用しない。

## v4 既知 TVT direct fit

ユーザー指示により、source / cluster `a,b` を選ぶ経路ではなく、query/test well 自身の known `TVT_input` だけで `dTVT ~= a*dZ+b` を fit する `known_tvt_fit_full` を追加し、selected にする。

- fit source: `TVT_input` が存在する全 known rows。unknown suffix の真値 `TVT` は使わない。
- fallback: known rows が不足して fit 不能な well だけ train source full-fit `a,b` の median に fallback する。
- prediction: last known `TVT_input` を anchor にし、unknown suffix の各 row で `TVT_next = TVT_prev + a*dZ + b` を累積する。
- 目的: 「test known TVT で fit し、それを未知 test に transform する」形式そのものの CV/LB を確認する。
