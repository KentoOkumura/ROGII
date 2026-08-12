# 設計

## 仮説

exp512で50%成分として使う完全な`hjyact_v2_final`を単独でhidden-safe再生成すれば、公開sourceの
Public LB `6.568`を再現でき、exp512の結果を公開componentの再現差とexp413とのblend効果へ分解できる。

## アプローチ

exp512のcompact self-contained inference候補を構成参照元にし、そこから完全な
`hjyact_v2_final`生成に必要なsource cellとruntime helperだけを抽出する。exp512のensemble orchestration、
exp413 runtime、cross-consumer candidate reuse、最終50/50式は取り込まない。公開componentを生成した後、
dynamic `sample_submission.csv`へone-to-oneで整列し、その値を変更せず`submission.csv`へ書く。

実装は新しいJupytext percent形式の
`exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py`から始める。
正規`*_inference.ipynb`はtemplate placeholderのまま保持し、候補の静的検証と採用承認後だけ置き換える。
train処理は存在しない。

## 実験範囲

- 対象実験: `exp513_hjyact_v2_final_standalone_public_lb_audit`
- Route: `ensemble`
- 親実験: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- 比較対象: hjyact source Public LB `6.568`、exp413 / exp510 Public LB `7.201`、進行中exp512 50/50 version 1。
- 変更する変数: exp512 finalを50/50 blendから`hjyact_v2_final`単独へ変更する。
- 固定する変数: source kernel version/run、active final path、profile、SP45/learned比、visible-prefix、
  model-package guard、seed-branch hedge、source model/input inventory、write順。
- 除外する経路: exp413 75 saved models、exp512 candidate reuse DAG、component CSV blend、weight arithmetic。

## Notebook設計

候補Notebookは次の7章で実装する。runtime helperとsource identity/input auditは同じ初期章へ統合し、
exp413 runtimeとblend章は持たない。

1. Imports, source identity, and mount-safe input audit
2. SP45 PF / Beam selector helpers
3. Ridge/PF anchor and deterministic candidate surface
4. Saved ridge artifact inference and runtime Ridge
5. Projection and learned trajectory replay
6. Guarded overlap and final hjyact-v2 layers
7. Standalone submission and reproducibility outputs

Notebook上でroute、parent、source version/run、final境界、variant数、model inventory、dynamic row/well数、
生成物パスを表示する。`from helper import main; main()`だけの薄い構成にはしない。同一実験helper importは
使わず、exp512候補から実験遂行に必要な関数・定数だけを抽出する。`__file__`は使わない。

## 固定component契約

- `SP45`と`learned trajectory`を`0.60 / 0.40`で結合する。
- guarded overlap overrideをsource順に適用する。
- visible-prefixは`balanced` profile、cut fractions `[0.50, 0.65, 0.75]`、calibration 24 seeds、
  final 48 seeds、350 particlesを固定する。
- model-package guardはmax weight `0.00425`、scale `6.0`、diff-p95 disable `25.0`を固定する。
- PF seed-branch hedgeはstrength `0.60`、min mass `0.25`、separation `[4.0, 40.0]`、cap `2.0 ft`を固定する。
- この処理後の`hjyact_v2_final`をそのままsubmissionへ書き、exp413や追加postprocessを適用しない。

## exp512 failure guard

- competition data rootは旧/current mount候補から`train`、`test`、`sample_submission.csv`の存在で一意解決する。
- Ridge rootは`data/train.csv`と5 trainer wrapperの存在・SHAで一意解決し、同じ監査済みrootを
  `RIDGE_ARTIFACT_ROOT`と`CFG.artifacts_path`へ明示的に渡す。
- source生成物に旧competition/Ridge rootの直接assignmentがないことを契約testで固定する。
- exp413 runtime / shared DAGを除外してcandidate sourceを236,961 bytesに抑える。package時はembedded
  bootstrapを含むNotebook全体がKaggle 1 MiB source制限内か再確認する。

## 段階gate

### Stage A: implementation / static validation

- source Notebook / code-cell SHAとmodel/input inventoryを固定する。
- exp512の公開componentだけが含まれ、exp413、50/50式、shared-DAGがないことをAST/契約testで確認する。
- Jupytext round-trip、`py_compile`、Ruff F821、専用test、`validate-exp`を通す。
- 実装完了後も正規Notebook採用、package、runは別承認とする。

### Stage B: visible source parity

- Kaggle GPU / internet offでdynamic sampleから生成する。
- ID one-to-one、sample order、finite、duplicate 0を確認する。static/precomputed predictionと
  inference-time training fallbackは0とし、source-defined defensive fallbackは変更せず実行logへ記録する。
- dynamic sample identityが既知visible sampleと一致した場合だけ、source final SHA
  `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`をpost-hoc assertionに使う。
- parity不一致ならfail-closeし、提出へ進まない。profile/threshold/seed/particle/write順で救済しない。

### Stage C: reproducibility

- 同じcanonical kernel id、GPU、internet-off、input versionで2回実行する。
- component prediction content SHAとsubmission SHAが一致することを必須とする。
- 一致前はdeterministic anchorと表記しない。

### Stage D: submit-check / Public LB

- output実ファイルが必要なため、この段階だけKaggle outputを取得する。
- `kaggle-submit-check`でsample互換、row数、ID、duplicate、missing、finite、予測統計、SHAを確認する。
- 別承認後にcode submissionを1回だけ行う。
- source `6.568`との表示精度一致をreproduction判定とし、exp413 / exp510 `7.201`との差を記録する。
- 結果にかかわらず同一実験内で再調整・再提出しない。

## 実行量設計

- active scientific variant: 1
- LightGBM train configs / new boosters / parent-control retraining: `0 / 0 / 0`
- source Ridge runtime fit: `1 config × 5 folds = 5`
- saved model files / contained estimators: `13 / 33`
- original runtime reference: GPU `787.7964199`秒
- hidden well数に応じたPF/Beam run数、peak RSS、実runtimeはpush前と実行後に記録する。

## 再現性設計

- seed policy: hjyact v2 source semanticsを保持する。global RNGやthread schedulingに由来する説明不能差分を許容しない。
- stochastic処理: SP45/learned likelihood-PF、visible-prefix seed bank、PF seed-branch hedge。
- PF/Beam / likelihood-PF / seed bagging: あり。sourceのseed数、particle数、分岐順を固定する。
- 並列処理と乱数: well単位の出力をkey/order付きで検証し、並列順序だけでSHAが変わる場合はfail-closeする。
- CPU/GPU runtime: GPU、internet off。source container digestとKaggle kernel versionを記録する。
- feature SHA: input、schema、row/well/feature count、decompressed content SHAを記録する。
- model / prediction / submission SHA: 13 model file inventory、model manifest SHA、component content SHA、
  submission SHA、rerun間byte/content一致を記録する。
- Kaggle bootstrap: prepare後にmetadataとbootstrap内config/source/model manifestをreadbackし、正規ファイルと照合する。

## リスク

- リークリスク: sourceのguarded overlap / visible-prefixはhidden-safe条件を個別監査する。visible output copy、
  static sidecar、well ID/cardinality分岐は禁止する。既知SHAは予測後のassertion専用とする。
- CV/LB不一致: honest OOFがなくPublic LBだけを測るため、LB改善をprivate一般化の証拠にしない。
- runtime / memory: sourceはGPU約788秒だが、hidden sample cardinalityで増える。exp413を除く一方、
  PF/Beamと5 Ridge fitsは残る。push前に動的run数とKaggle制限を再評価する。
- 再現性: stochastic PFとGPU/container差でSHAが揺れる可能性がある。2 rerun不一致ならanchor化も提出も行わない。
- source drift: version 2 / run `337064157`と保存SHAを固定し、最新公開Notebookへ暗黙更新しない。
- 提出回数: 1回だけ使用する。technical gate未達の提出やLB後の探索に提出枠を使わない。
