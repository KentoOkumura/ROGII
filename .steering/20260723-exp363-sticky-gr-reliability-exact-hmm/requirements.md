# 要件

## 依頼

HMM の状態変数案の1つとして、sticky GR reliability exact HMM の backlog、
実験ディレクトリ、steering を作り、実装前の設計を確定する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 初回設計タスクでは実装、Notebook置換、Kaggle実行、推論、提出を行わない。
- 2026-07-23 の後続依頼 `exp363を実装してください` により、Stage 0 の実装と
  placeholder Notebook の置換だけを追加承認する。Kaggle push/run、Stage 1、
  推論、提出は引き続き未承認とする。
- 2026-07-24 の後続依頼 `実行してください` により、固定済みStage 0の
  Kaggle CPU package / push / runだけを追加承認する。Stage 1、推論、提出には
  拡張しない。
- exp209 の rate transition、grid、emission sigma、posterior outputを固定する。
- rate / rate change の prefix・geometry予測を前提にしない。
- 同一OOFで係数・遷移確率を探索しない。

## 受け入れ基準

- q の状態、遷移、emission作用が一意に固定されている。
- Stage 0 / Stage 1 の入力、truth freeze境界、全gate、実行量、fail policyが固定されている。
- 1 variant / 773 HMM runs / 0 booster / parent control rerun 0が記録されている。
- backlog と experiment summary に設計確定・未実装として現れる。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
- Stage 0 は diagnostic 1 / reporting folds 5 / HMM well-run 0 / model config 0 /
  trained fold 0 / booster 0 / parent control rerun 0として実装される。
- Stage 0 の block tail、posterior集約、circular offset、fold pass の低レベル契約が
  config とテストで一意に固定されている。
- Stage 0固定AND gateを1つでもFAILした場合は、同じbranchで救済せず閉じる。
