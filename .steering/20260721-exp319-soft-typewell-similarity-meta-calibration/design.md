# 設計

## アプローチ

Type Well GRをnative TVT gridへ正規化し、quantile 5、autocorr 3、gradient MAD、missing fractionのdescriptorをouter-train robust scaleする。diagonal Mahalanobisで近傍3群を選び、`exp(-distance)`重みでexp311 priorを平均する。nearest distanceがouter-train p90を超えればglobal priorへfallbackする。

## 実験範囲

- 対象: `exp319_soft_typewell_similarity_meta_calibration`
- Route: `pf_beam`
- 親: `exp313_typewell_group_unseen_transfer_guard`
- 変更: unseen/singleton向けsoft fallbackだけ。
- 固定: descriptors、metric、k、kernel、temperature、cutoff、permutation control。
- 計算量: scientific 1 + control 1、5 folds、model/booster/decoder 0。

## 再現性設計

- descriptor schema、robust scaler、group content SHA、neighbor table、tie-breakを保存する。
- 距離tieはgroup content SHAの辞書順で決定する。
- descriptor permutationはfold keyからstable SHA256順を使う。

## リスクと停止条件

- content similarityが地質的な校正共通性を表さない可能性がある。
- far neighborへ無理に転送せずglobal fallbackを固定する。
- permutationとの差またはgroup-out安定性がFAILならsoft pathをexp313へ追加しない。
