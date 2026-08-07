# 設計

## アプローチ

exp238 の候補集合とnested selector構造を固定し、contextだけをinference-safeにする。

1. `copcf_*` 41列は exp109/114 OOF row ID と exp065/114 train-well geometry に依存し、
   hidden test変換が実装されていない。値埋めではなく、selector学習schemaから除外する。
2. exp226診断4列はアルゴリズム上current testでも計算済みだが、既存推論が最終`tvt`しか
   保存していなかった。exp245 inferenceで`PredictionResult.geop`と`delta`を行単位に展開し、
   `exp226_geop_tvt`、`exp226_gr_delta`、差分、絶対差分を生成する。
3. selector train summaryのcontext schemaを正とし、inferenceで欠損列を`NaN`追加するfallbackを
   廃止する。missing列または非有限値が1件でもあればfail-fastする。
4. exp245の結果がguardを通過した後だけ、別実験でselector top1 pathの直接採用を評価する。

## 実験範囲

- 対象実験: `exp245_selector_context_parity_on_exp238`
- Route: `ensemble`
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 変更する変数: selector context schema、exp226 current-test診断列生成、inference parity guard
- 固定する変数: 11候補、outer5/inner4 split、candidate-error objective、学習上限、selector LightGBM params、安全性指標

## 再現性設計

- seed policy: exp238と同じseed 42。bounded selector row samplingはouter/inner fold由来の固定seedを使う。
- stochastic 処理の有無: selector samplingとraw-test PF replayあり。global RNGは使わず親のstable per-well seed契約を継承する。
- PF/Beam / likelihood-PF / seed bagging の有無: candidate再生成にexp218 replayを使う。selector学習では固定済みgroup-safe OOFを読む。
- 並列処理と乱数の関係: replayはwell ID由来stable seed、selectorは`default_rng(seed)`で固定する。
- CPU/GPU runtime と deterministic flags: selector train/inferenceともCPU。GPU final trainはexp245では実行しない。
- train cache / test feature regeneration の SHA 記録方針: context schema SHA、nested scoreのdecompressed SHA、current-test selector surfaceのdecompressed SHAを保存する。
- model manifest / prediction / submission SHA 記録方針: 20 selector modelのSHAとouter/inner完全被覆を保存する。exp245ではsubmissionを作らない。
- Kaggle package bootstrap 確認方針: exp237 helper/config、exp218 replay source、exp226 sourceがbootstrapに含まれ、loose fileと一致することを確認する。

## リスク

- リークリスク: exp226 OOFはouter-valid wellから隔離された固定入力、selectorはstrict outer/inner cross-fitを維持する。
- CV/LB 不一致リスク: 41 OOF-only signalを除くためselector CVが悪化する可能性がある。一方、hidden testとのschema差は解消する。
- ランタイム/メモリリスク: 20 CPU boostersとcurrent-test replay/exp226再生成。context削減によりexp238よりメモリは減る見込み。
- 再現性リスク: current-test PF replayを含むため、SHA一致を確認するまではdeterministic submission anchorと呼ばない。
