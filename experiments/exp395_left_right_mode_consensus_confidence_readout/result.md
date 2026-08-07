# exp395_left_right_mode_consensus_confidence_readout 結果

## 仮説

同じstable HMM mode lineageに対するheel側 / toe側のdisjoint GR evidenceが
異なるmodeを支持する区間ほど、物理モデルのpersistent offsetとlarge errorが多い。

## 設定

- 親: `exp391_prefix_anchored_mode_persistence_hmm_readout`
- decoder: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 検証: outer 5-fold、`well_id` group、truth-late confidence readout
- primary metric: `abs(error) > 10 ft` AUC
- reporting metric: RMSE
- シード: RNGなし、stable order

## 結果

前提のexp391 Stage A1がtechnical / mechanism gateをFAILしたため、固定契約に従い
Notebook/test実装を開始せず閉じた。Stage 0、full OOF、Kaggle runは未実行。

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false
- seed policy: RNGなし、固定key順
- kernel version: 未実行
- input / confidence content SHA: 未生成
- decoder contract SHA: 未生成
- prediction SHA: prediction生成なし
- submission SHA: submission無効
- rerun result: 未実行

## 解釈

設計だけを確定した。exp386 scenario bankは生成されずexp387も閉鎖済みのため、
現時点で利用可能なexp391 stable mode lineageをmode carrierに選んだ。

exp391 Stage A1のHMM支持は1/19 events・1/5 foldsで、必要な60%以上・4/5 foldsを
満たさなかった。parity、normalization、runtimeもFAILしたため、mode carrierの
先行条件は不成立。事前のgo/no-go契約どおりexp395を未実装で閉じる。

## 次

実装、Stage 0、full OOF、予測補正、inference、submissionへ進まない。
