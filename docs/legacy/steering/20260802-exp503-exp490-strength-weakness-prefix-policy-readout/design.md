# 設計

## アプローチ

保存済みexp490 OOFを1回読み、行ごとの `candidate-parent` 補正量、両者の誤差、
suffix深度、posterior標準偏差を作る。そこからwell単位に以下を集約する。

1. RMSE/MAE/bias/error drift、累積SSE gain、最長連続悪化、worst 128-row window。
2. suffix絶対深度（0,128,256,512,1024,2048,4096+）と相対四分位のgain curve。
3. exp499の凍結済みtarget-free 32特徴とのSpearman、beneficial-well AUC、quantile lift。
4. depth gain curveを固定seed KMeansで5 archetypeに分け、代表wellを可視化する。

公開notebook `fle3n-rogii-v5` のfade-in
`1-exp(-md_since/tau)` を、
`parent + alpha * ramp * (exp490-parent)` として再生する。`alpha=0` と、
`alpha={0.25,0.5,0.75,1.0}` × `tau={0,50,85,125,250,500,1000}` の29 profileを
固定する。各outer foldについて残り4 foldsのpooled RMSEだけで1 profileを選び、
held foldを評価する。

さらに固定depth-2 CARTでwell-optimal alphaを推定する。strict prefix版は
`visible_prefix_rows/prefix_gr_sigma/prefix_gr_information_ratio`だけ、target-free
context版は事前登録したexp499特徴を使う。treeはouter 4 foldsだけでfitし、held-wellへ
適用する。これは説明可能性・適用可能性の監査であり、submission候補の自動承認には使わない。

公開notebookのmasked-prefix backtestを完全再生するには各cutoffからexp490 HMMを再実行する
必要があるため、本expでは行わない。代わりにearly 128/256/512 suffix rowsの正解でprofileを
選び、その後半だけを評価する楽観的transfer auditを置く。これでも改善しない場合、
追加HMM replayの優先度を下げる。

## 実験範囲

- 対象実験: `exp503_exp490_strength_weakness_prefix_policy_readout`
- Route: `ensemble`
- 親実験: `exp490_geometry_centered_mean_reverting_offset_hmm`
- 比較対象: 保存済み`exp357_parent_prediction`
- 変更する変数: readout slice、fade profile、outer-fold-safe alpha policy。
- 固定する変数: upstream予測、fold、truth、候補生成、HMM/PF/Beam、raw feature生成。

## 再現性設計

- seed policy: `random_state=42`固定。KMeansのみstochasticで`n_init=20`固定。
- stochastic 処理の有無: KMeansだけ。CART、集計、固定gridはdeterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: 全て0。保存済み予測だけを読む。
- 並列処理と乱数の関係: 単一process、`n_jobs`並列なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU無効。
- train cache / test feature regeneration の SHA 記録方針: exp490 gzipはrawとdecompressed
  SHA、exp499 feature CSVはfile SHA、出力CSV/JSON/plotはfile SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: tree policy manifestとcross-fitted
  policy prediction CSVのSHAを記録する。submissionは生成しないためnot applicable。
- Kaggle package bootstrap 確認方針: prepare後のmetadataと埋込configをstrict validationで確認する。
- Kaggle package bootstrap 確認方針: strict prepare後にmetadata、CPU/internet、2 kernel
  sources、package内contract testを確認し、同じcanonical IDへversion追加する。

## リスク

- リークリスク: truth-aware特徴関連・archetype・early-truth transferは説明専用。
  deployability判断はouter-held policyだけに限定する。
- CV/LB 不一致リスク: train OOF well分布だけの結論でありhidden testへ直接移植しない。
- ランタイム/メモリリスク: 330 MiB gzipを全件DataFrame化すると約1.5--2 GiB。
  Kaggle CPU RAM内に収め、不要列を読まない。
- 再現性リスク: pandas/sklearn version差でKMeansラベル順が変わり得るため、clusterは
  強弱順位の根拠ではなく記述に限定し、入力/出力SHAを残す。
