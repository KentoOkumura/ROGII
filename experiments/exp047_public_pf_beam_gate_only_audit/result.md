# exp047_public_pf_beam_gate_only_audit 結果

## 仮説

PF/Beam を直接候補や自由な残差補正として使うのではなく、`base + w * (candidate - base)` の保守的な重み調整に限定すれば、exp031/033/035/045 のような大きな Public LB 悪化を避けながら PF/Beam の有効行だけを利用できる可能性がある。

## 設定

- 親: `exp046_hidden_branch_surrogate_audit`
- 入力: `exp029_public_sel15_pf_oof_feature_generation` の PF/Beam 生成物。train well の途中以降を隠し、本番 test 風に予測させたもの。
- 監査 split: original-fold、well-hash、stratified-group fold
- reference: `exp026_pseudo_tail_bucket_shrink`
- 候補: fixed gate、learned optimal-weight gate、learned PF-wins gate
- 制約: すべて `base + w * (candidate - base)`、`w <= 0.40`

## 結果

Kaggle train version 1 で full audit が完了した。

| メトリック | 値 |
| --- | --- |
| rows / wells | 1,782,279 / 773 |
| original-fold best | 14.527279 (`exp026_to_pf_gate_w0p30`) |
| well-hash best | 14.620835 (`exp026_to_pf_gate_w0p30`) |
| stratified-group best | 14.353489 (`exp026_to_pf_gate_w0p30`) |
| original-fold delta vs exp026 / public PF / pf090 | -1.116453 / -0.645357 / -0.562252 |
| well-hash delta vs exp026 / public PF / pf090 | -1.205268 / -0.551801 / -0.468697 |
| stratified delta vs exp026 / public PF / pf090 | -1.118336 / -0.819147 / -0.736043 |
| distance buckets | all buckets improved vs exp026 anchor |
| best learned gate | 14.573792 (`learned_pf_gate_ridge_wmax0p40`, original-fold) |
| validation | PASS |
| tests | 10 passed |
| Kaggle train | version 1 COMPLETE |

## 解釈

固定 `exp026_to_pf_gate_w0p30` が original-fold、well-hash、stratified-group の全てで最良だった。距離 bucket でも 0-49 rows から 2500+ rows まで全 bucket で exp026 anchor より改善しており、近距離 bucket を壊す兆候はこの代理面では出ていない。

learned gate は `learned_pf_gate_ridge_wmax0p40` が全 split で 2 位だったが、固定 `w=0.30` より弱い。現時点では learned gate の複雑さを採用する根拠はない。

ただし、exp046 で確認した通り、train-side surrogate の改善は Public LB への転移を保証しない。exp031 の固定 `pf090_hold010` は surrogate で改善しても Public LB 8.956 と exp027 8.781 から悪化している。したがって、この結果は「固定 gate 候補を次の inference port audit に進める根拠」であり、即 code submit の根拠ではない。

## 次

`exp027` Public LB 8.781 を維持する。次に進めるなら、別実験で固定 `exp026_to_pf_gate_w0p30` だけを inference port し、output diff、予測範囲、public sample blind spot を確認してから code submit 可否を判断する。
