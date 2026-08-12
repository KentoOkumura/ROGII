# exp341 missing-gap calibrated soft variance exp226 residual HMM

## 状態

- Route: `pf_beam`
- 状態: 設計確定のまま未実装・未実行、exp339 dependency FAILで閉鎖
- 優先度: P2 conditional
- 親実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`

## 仮説

exp339で校正した補間誤差分散を、元データでGR欠損だった行だけ基礎分散へ加算すれば、観測を完全に捨てたexp269より穏当な形で過信を抑えられる。

## 検証方針

- 観測行はexp281とbit-level相当の尤度に保つ。
- 欠損行だけ `sigma_eff^2 = sigma_exp281^2 + sigma_imp^2` とする。
- 補間GR、状態、遷移、候補bank、欠損判定、出力はexp281から変更しない。
- exp339全gate通過とfold別分散表SHA凍結を実装開始条件とする。
- 実行時は新規variant 1個、773 well HMM run。既存controlは再学習・再実行しない。

## 所見

exp339のreal-vs-circular fold gateが2/5でFAILし、補間分散表を自然欠損へ転送する根拠が不足した。soft variance HMMへ進めない。

## 実装境界

notebookはscaffold placeholderのまま保持する。FAIL tableを使った実装・実行、救済grid、inference、submissionは行わない。

## 文書

- Steering: `../../docs/legacy/steering/20260722-exp341-missing-gap-calibrated-soft-variance-exp226-residual-hmm/`
- 設定: `config.yaml`
