# 要件

## 依頼

PF粒子をbaseと6地層modeへ層別化し、resamplingでformation-relative仮説が消えるのを防ぎながら物理候補を利用する。今回は設計のみ確定する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp377の識別可能性とexp378の候補新規性が合格するまで実装しない。
- 総粒子数600、初期配分base 300・各formation 50を固定する。
- resample後の最低配分base 150・各formation 25、残り300をposterior比例とする。
- Stage 0はseed 0だけ。4 seed full runは別途ユーザー承認を要する。
- 親controlを再実行しない。

## 受け入れ基準

- Stage 0でexp271 seed0より0.10 ft以上改善、5 fold中4 fold以上正、scope悪化0.02 ft以下である。
- mode survivalが99%以上、ESS/粒子数のp05が0.05以上、H512 novelty gainが0.05 ft以上である。
- Stage 1 mean4でexp271 mean4より0.10 ft以上改善、5 fold中4 fold以上正、p95悪化0、worst悪化0.25 ft以下である。
- Stage 1のH512/whole novelty gainが各0.05 ft以上である。
- 同一wellの単独・並列・順序変更実行が一致し、stable SHA seedとPF診断SHAが記録される。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
