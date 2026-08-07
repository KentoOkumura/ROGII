# exp319_soft_typewell_similarity_meta_calibration

## 状態

- ルート: `pf_beam`
- 状態: 設計確定・exp311/313待ち・未実装
- 親: `exp313_typewell_group_unseen_transfer_guard`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

exact群peerがないwellでもType Well content descriptorが近い群のcalibration/noise priorはsoft transferできる。exact pathは置換せず、unseen/singleton fallbackだけを対象にする。

## 検証方針

leave-one-group-out nested CVでreal/permuted、far fallback、worstを評価する。全gate PASS時だけexp313 fallbackへの採用を提案する。

## 所見

soft transferはexact groupの代替ではなく、unseen/singleton時の条件付きfallbackだけにする。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`でKaggle package/push/runは禁止。
