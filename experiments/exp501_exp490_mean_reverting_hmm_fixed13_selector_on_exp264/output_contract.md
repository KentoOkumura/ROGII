# exp501 出力契約

## 実装済み、未実行の Stage A / C 出力

Kaggle CPU 実行を別途承認した場合に限り、次を生成する。

- exp490 gzip raw / decompressed SHA、allowlist、global key、suffix offset、fold repartition監査
- raw-test-safe feature catalog / schema / content manifest
- 40 selector model manifest
- 25 nested compact partition manifest
- 49,191,857行のouter-valid candidate score
- dual selector score metrics
- fixed13 / parent fixed12のpaired scope・fold・by-well metrics
- exp490 candidate usageとincumbent reranking診断
- post-freeze H512 / whole-well add-one oracle診断
- scientific AND gate、summary、reproducibility manifest

現時点ではcompact Jupytext候補と契約テストだけを実装済みであり、上記生成物は未生成である。

## 出力しないもの

- 新しいHMM / PF / Beam prediction
- exp498 physics featureまたはexp499 well-router feature
- parent exp264 controlの再学習
- downstream TVT model / OOF
- current-test prediction
- `submission.csv`

## 判定

technical、leakage、selector score、利用率、parent fixed12対比のpooled / fold / scope / by-well
tailを全ANDで判定する。FAIL時は
`FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`で閉じ、同一OOFで救済しない。
PASSしてもcurrent-test、inference、submissionへ自動昇格しない。
