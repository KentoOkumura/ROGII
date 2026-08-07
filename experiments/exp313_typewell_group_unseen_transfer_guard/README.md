# exp313_typewell_group_unseen_transfer_guard

## 状態

- ルート: `pf_beam`
- 状態: 設計確定・exp311/312待ち・未実装
- 親: `exp312_typewell_group_conditional_gr_emission_table`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

同群peer/supportのavailabilityとidentity fallbackをtruth-freeに固定すれば、Type Well群priorを未知群へ誤転送するriskを分離できる。校正値は変更せず、後続全体の共通safety gateだけを作る。

## 検証方針

same-group、leave-group-out、spatial+typewell purgeの3面でcoverage、identity parity、same-group gain、unseen non-regression、worst guardを評価する。本exp FAILならexp314〜320はblockedのまま。

## 所見

testのType Well群構成が変わっても安全側へ落ちることを、下流実装より先に保証する。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`でKaggle package/push/runは禁止。
