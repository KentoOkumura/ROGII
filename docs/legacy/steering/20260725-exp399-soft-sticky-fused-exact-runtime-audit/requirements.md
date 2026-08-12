# 要件

## 依頼

exp394 soft-sticky E/H exact HMMについて、TVT grid約573点とresidual-rate 41状態を
一切削減せず、同じ数式とposterior readoutを保ったまま実行時間を短縮する。
具体的には、遷移先配列の毎row materializationを除去し、H→E dockingを疎な
H→H遷移計算へ融合し、well単位の2並列を導入する。

## 制約

- Route: `pf_beam`
- 親: `exp394_soft_sticky_exp226_k16_branch_hmm`
- scientific variantは1つだけとし、soft-sticky、K16 schedule、emission、grid、
  rate states、posterior mean、switching/docking式を変更しない。
- `retain_all_tvt_grid_states=true`、`retain_all_rate_states=true`を維持する。
- 親fixed16 prediction / branch posterior / scheduleは保存生成物をload-onlyで参照し、
  親controlを再実行しない。
- Stage 0は固定16 wellsのtechnical auditだけとし、suffix truth、RMSE、exp263、
  hidden-like roleを読まない。
- candidate 1 / HMM well runs 16 / reporting folds 5 / LightGBM config 0 /
  trained fold 0 / booster 0 / parent control rerun 0 / GPU 0。
- full 773-well OOF、inference、submissionはStage 0 PASS後の別承認対象とする。
- 再現性: `docs/06_reproducibility.md` に従い、並列後もwell順、row順、状態順、
  output row順を固定し、入力・source・prediction・posterior・scheduleのSHAを記録する。

## 受け入れ基準

- fixed16のwell / row identityが親と完全一致する。
- candidate prediction、`gamma_E/H`、H conditional mean/std、joint std、switch/docking
  readoutが親fixed16に対する事前固定の数値許容差内にある。
- posterior normalization max error `<=1e-8`、transition row-sum max error `<=1e-10`、
  finite coverage / H full-grid coverageが各`1.0`。
- parent fixed16比のstate-time normalized speedupが`>=3.684212x`かつ、
  15% safety margin込みfull runtime projectionが`<=30,600 sec`。
- fixed16 projected peak RSSが`<=25 GB`。
- 2 wells並列でも出力はstable sortされ、乱数とthread schedulingに依存しない。
- deterministic anchor として扱う場合は、input/source/prediction/posterior/scheduleの
  content SHAとKaggle kernel versionを記録する。model / submissionは存在しない。
- gzip生成物を比較する場合はraw gzip SHAではなくlogical/decompressed content SHAを
  主証拠として記録する。
