# exp496 出力契約

## Stage A / Cで生成したもの

- 入力・SHA・global key・fold repartition監査
- raw-test-safe feature catalog / schema / content manifest
- 40 selector model manifest
- 25 nested compact partition manifest
- 49,191,857行のouter-valid candidate score
- selector score metrics
- fixed13 / parent fixed12のpaired scope・fold・by-well metrics
- exp486 candidate usageとincumbent reranking診断
- post-freeze H512 / whole-well add-one oracle診断
- scientific AND gate、summary、reproducibility manifest

Kaggle private CPU version 1（id_no `129287597`）で全項目を生成した。
巨大なouter-valid score parquetと25 compact partitionsはKaggle側へ保持し、
判定、再現性、後続readoutに必要な小さいmetrics / manifestと完全logsだけを
`kaggle/output/train_v1/`へ選択取得した。

## 出力しないもの

- 新しいPF / HMM / Beam prediction
- exp486 Residual版やHMM blend
- downstream TVT model / OOF
- current-test prediction
- `submission.csv`

## 判定

technical、leakage、selector score、利用率、parent fixed12対比の
pooled / fold / scopeはPASSしたが、by-well p95 / worst gateをFAILした。
事前契約どおり`FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`で閉じ、同じOOFで
救済しない。downstream、current-test生成、inference、submissionは出力しない。
