# 設計

## アプローチ

exp238 selector train v4 の184 context schemaと保存済み20 modelを固定する。current testでは
exp218 replay、HMM、exp226候補・診断、multiobs、enrichmentに加え、次のfull-train prior
変換を実行する。

1. train/test typewell GR のnative overlapをtest wellごとに計算し、test wellを固定済み
   train clusterへ割り当てる。test-test edgeは使わない。
2. typewell priorは割当先clusterのfull train wellの真のanchor差分をtest `md_since`へ補間する。
3. spatial priorはtrain geometryだけで標準化し、test軌跡からtrain近傍を選び、同じ差分を補間する。
4. train cluster center/scaleとtrain well近傍からcluster outlier/gate特徴を作る。
5. exp237と同じ命名・派生式で41個の`copcf_*`を生成し、保存済みselectorへ渡す。

test targetは一切読まない。full train targetは本番推論で利用可能な学習済みreference priorの
構築だけに使用する。

## 実験範囲

- 対象実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- Route: `ml_model`
- 親実験: exp238 selector train v4 / final train v5
- 変更する変数: raw-test selector context generatorだけ
- 固定する変数: 184 context schema、41 copcf列、11 candidate列、outer5/inner4、20 selector model、selector/final model parameter、fold/model SHA

## 再現性設計

- seed policy: exp218 replayのstable per-well seedを継承し、prior生成は乱数を使わない。
- stochastic 処理の有無: prior/cluster/typewell割当はdeterministic。候補replayの乱数だけ親契約を継承する。
- PF/Beam / likelihood-PF / seed bagging の有無: current-test候補再生成として存在するが、新規seed探索はしない。
- 並列処理と乱数の関係: prior生成はsingle-processで、test wellをsorted orderに処理する。
- CPU/GPU runtime と deterministic flags: parity auditはCPU。selector/finalの学習は0。
- train cache / test feature regeneration の SHA 記録方針: train cache、cluster assignment、train geometryのinput SHAと、test context schema/content decompressed SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 20 model SHAを再検証し、5 score面のdecompressed SHAを記録する。submissionは作らない。
- Kaggle package bootstrap 確認方針: prepare後にconfig、exp218 replay、exp226 source、親configの同梱とCPU/internet offを確認する。

## リスク

- リークリスク: test targetは使わない。train targetはfull-train reference priorだけに使う。test同士の近傍化は禁止する。
- CV/LB 不一致リスク: trainはfold-safe OOF prior、testはfull-train priorなのでcoverage/分布が変わる。列定義、coverage、quantileを保存して監査する。
- ランタイム/メモリリスク: exp218 replayとexp226 full-train fitに加え、773 train wellのprior補間が必要。row全組合せは作らずwell単位で処理する。
- 再現性リスク: hidden well数で結果が変わらないようtest-test距離・test-test cluster edgeを禁止する。rerun SHA一致前はdeterministic anchorと呼ばない。

## Phase 2接続設計

parity notebookのcurrent-test selector context生成を正とする。context生成後にexp145 learned
likelihood、exp218 U projection / GRWRをcurrent testから再生成し、exp218 380特徴を組み立てる。
outerごとのselector score計算時に35 rank-slot特徴を作り、同じouter foldに属する保存済み
3 final LightGBMへ渡す。全15予測を等重み平均する。

- selector source: exp238 selector train v4、outer 5 × inner 4 = 20 saved models
- final source: exp238 final train v5、outer 5 × 3 configs = 15 saved models
- 学習: selector 0、final 0、control 0
- runtime: T4 packageを維持するが、GPU学習は行わない
- input kernel sources: 既存hidden-safe inference 9件に、full-train `copcf_*`生成用の
  exp065 / exp109 / exp114を追加する
- public-test parity検証: 184 context、41 copcf、exp226診断4、380+35 schema、15 model、
  sample ID順、prediction/submission finiteをfail-fastする
- 既存v3とのprediction差は記録するが、同一submissionになることは要求しない。旧v3は45列を
  NaN routingしていたため、差が生じるのが想定される
