# exp385_gr_typewell_likelihood_on_vector_drift_paths 結果

## 状態

exp383 Stage 0 resource FAILにより、設計のみ・未実装・未実行で閉鎖。

## 仮説

horizontal GRとtypewell GRのfixed likelihoodは、exp384の複数物理pathを
target truthなしで識別し、exp384を0.50 ft以上改善できる。

## 設定

- 親: exp384
- 検証: exp384と同じouter 5-fold
- Stage 0: known-prefix/circular likelihood readout、full decoder 0
- Stage 1: 別承認時のみ773 exact forward-backward runs
- fitted model / PF / Beam / booster: 0

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: まだ扱わない
- kernel / SHA / rerun: 未実行

## 解釈

性能判断はまだない。candidate pathを変更せず、GR/typewellの識別力だけを切り分ける。

## 次

exp383/384依存が成立しないため実装しない。
