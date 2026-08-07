# exp379_formation_transition_mixture_exact_hmm セッションノート

## 目的

formation-relative rateをexact HMMのlatent transition modeとして利用する。

## 現在の状態

- Route: pf_beam
- 状態: exp377 Stage 1 scientific FAILにより未実装のまま終了
- CV: まだなし
- LB: まだなし

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp379 TITLE=formation-transition-mixture-exact-hmm
make new-exp EXP=exp379 SLUG=formation_transition_mixture_exact_hmm
```

## 変更点

- 7 mode、区間内固定、境界stay 0.95を固定。
- Stage 0=16 wells、Stage 1=773 wells。実行なし。

## 再現性メモ

- seed policy: no RNG exact HMM
- stochastic components: なし
- CPU/GPU runtime: CPU設計、未実行
- Kaggle kernel id / version: なし
- input / feature schema SHA: 未生成
- feature content SHA: 未生成
- model manifest / model SHA: 未生成
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun check: 未実行

## 次のアクション

1. 現設計を実装・実行しない。
2. 16坑井Stage 0や773 HMM runsを開始しない。

## 依存gate結果

2026-07-24、exp377 v2はStage 0をPASSしたが、truth-late Stage 1で
median6 pathがdirectより`22.676107 ft`悪化し、個別6面も全悪化した。
exp378も未実装のまま終了するため、HMM側でscientific negativeを救済せず現設計を閉じる。
