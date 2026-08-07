# exp384_fault_aware_piecewise_stratigraphic_vector_field 結果

## 状態

実装済みだが、exp383 Stage 0 resource FAILにより未実行で閉鎖。

## 仮説

exp383のsmooth fieldに対し、formation/structural discontinuityで分けた
piecewise fieldをsoft周辺化すると0.50 ft以上改善できる。

## 設定

- 親: exp383
- 検証: exp383と同じouter 5-fold
- metric: suffix row RMSE
- 予定量: 1 candidate / 5 reporting folds / model・HMM・PF・Beam・booster各0
- 実装検証: 専用test 14件、Ruff、py_compile、Jupytext round-trip、
  strict experiment validation PASS
- 全repository test: `853 passed / 6 skipped / 2 failed`。2件は未変更の
  exp296 config status/run-approval期待不一致で、exp384 testは全PASS

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

fault graphからlate scoringまでの正規Notebook実装は完了した。ただしexp383の
Stage 0/1 PASS生成物が存在せず、parent manifest SHAも未固定なので、性能判断はない。
この状態でKaggle runするとparent SHA gateでfail-closedになる。
2026-07-24にexp384のpackage / push / 実行承認を得たが、ローカルとKaggleの
双方にexp383 kernel・PASS生成物がないことを確認したため、無効なrunは開始していない。

## 次

exp383 PASS artifactが生成されないため、package/push/runを行わない。
