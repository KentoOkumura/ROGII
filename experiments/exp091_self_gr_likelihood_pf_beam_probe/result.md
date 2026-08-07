# exp091_self_gr_likelihood_pf_beam_probe 結果

## 状態

Kaggle train v1 完了。

## 仮説

既存 PF/Beam/likelihood-PF 候補に、同一 horizontal well の GR self-similarity 由来候補を追加すると、直接置換ではなく候補集合として真値近傍を含む bucket が増える可能性がある。

## 評価方針

`exp072_exp063_full_replay_feature_cache` の deterministic train cache を読み、以下を比較する。

- 候補別 RMSE / MAE / within 1, 2, 5, 10 ft coverage
- oracle topK coverage
- target-free `candidate_rank_score` topK coverage
- distance bucket / tail-rank bucket 別 miss rate
- by-well worst case

評価区間の true TVT は coverage 計算だけに使い、self-GR 候補生成や ranking score には使わない。

## 結果

- rows: 3,783,989
- wells: 773
- runtime: 1,533.57 sec
- source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`

候補単体では `likpf_mean` が最良で、RMSE 11.594897 / MAE 7.067633 / within 10ft 0.772807。PF ANCC は RMSE 14.493051、Beam mean は 15.774327。

self-GR 候補単体は弱い。`self_gr_ens` は RMSE 191.215912 / MAE 134.880295 / within 10ft 0.135627、`self_gr_best` は RMSE 250.161697 / within 10ft 0.270874。直接置換や単体 candidate としては不採用。

一方で oracle best candidate は RMSE 6.873199 / MAE 3.231436 / within 10ft 0.925153 で、候補集合としての headroom は大きい。oracle の selected self-GR rate は 0.135212 あり、self-GR candidate が一部 row では近傍候補を含む。

ただし現行の target-free `candidate_rank_score` は top1 RMSE 29.985529 / within 10ft 0.746819 で `likpf_mean` 単体より悪い。top10 まで見ると RMSE 6.953187 / within 10ft 0.922684 まで oracle に近づくが、これは候補集合を広く見た場合の headroom であり、実用 selector ではない。

## 解釈

self-GR path は candidate generator として全面的には失敗している。特に distance bucket 全域で `self_gr_ens` / `self_gr_best` の miss rate が高く、GR self-match だけで TVT path を作る方針は支持しない。

ただし oracle headroom と selected self-GR rate から、候補集合の中に局所的に有効な self-GR 候補が混ざる可能性は残る。次に進むなら、self-GR を直接採用するのではなく、PF/Beam/likPF 候補を主にした supervised candidate ranker で self-GR score を補助特徴として使う範囲に限定する。

## 次の判断

`self_gr_likelihood_pf_beam_probe` は診断完了。直接置換、hard switch、PF likelihood / beam pruning への即時投入はしない。後続は `pf_candidate_coverage_then_ranker_audit` または supervised ranker 小実験に統合し、self-GR は候補値そのものではなく ranker feature として扱う。
