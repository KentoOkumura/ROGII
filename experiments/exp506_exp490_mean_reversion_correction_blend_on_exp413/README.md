# exp506_exp490_mean_reversion_correction_blend_on_exp413

## 状態

- ルート: `ensemble`
- 状態: Stage A version 2 COMPLETE / primary gate FAIL / terminal close
- primary CV: `7.902068462119896`
- selected anchor CV: `7.884802794404715`（exp413 Stage D保存OOF）
- Public / Private LB: なし / なし
- inference / submission: なし / なし
- kernel: `kentookumura/exp506-exp490-mean-revert-correction-exp413-train` version 2

## 仮説

exp490の絶対予測ではなく、exp357からexp490へ変化した補正だけをexp413 anchorへ小さく加える。

```text
prediction = anchor + lambda * (exp490 - exp357)
```

lambdaは他4 foldsでclosed-form fitし、`[0.00, 0.10]`へ制限してheld foldへ適用した。
通常の`anchor / exp490` convex blendはreport-only、selectable=falseに固定した。

## anchor解決

exp497 Stage E version 1は`completed_gate_failed_closed`で終端したため、exp506 outcomeを見る前に
exp413 Stage D保存OOF（CV `7.884802794404715`、file SHA `9bd2d177...cef4a9d`）をanchorへ固定した。

## 検証方針

exp413と同じouter 5 foldsを使い、held foldを除く4 foldsだけでlambdaをfitする。
pooled gain、5/5 folds、MD 3面、hidden-like 2面、by-well tail、lambda安定性の全ANDで判定し、
1条件でもFAILならweight / component / scope / gateを救済せず終端する。

## Stage A結果

- primary CV: `7.902068462119896`、anchor比`+0.017265667715181 ft`悪化
- lambda: `[0, 0.041578388, 0, 0.004513714, 0]`、deployment中央値`0.0`
- fold nonworse: `3 / 5`
- fixed scope nonworse: `0 / 5`
- by-well delta p95 / worst: `+0.054729023 / +1.816049513 ft`
- technical / leakage / input SHA checks: 全PASS
- decision: `FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE`

report-only convex controlはCV `7.7345312772318815`だったが、全foldで上限0.10へ張り付いた
非選択診断であり、primaryを救済しない。

## 実行量

scientific primary 1、report-only control 1、outer/meta 5/5、model / booster / HMM / PF / Beam / GPU /
親再学習は全て0。入力は保存済みexp413、exp490、exp115 hidden-likeの3 notebook outputだけを使用した。

version 1は最後のmetrics表示のNumPy bool serializationでERROR。既存`to_jsonable()`を表示にも適用し、
回帰テストを追加したversion 2がprivate CPU / internet offで`294.943 sec`でCOMPLETEした。

## 所見

補正係数の符号はfold間で安定せず3/5が下限0となり、pooled・fold・全scope・worst-well・lambda安定性の
事前gateをFAILした。exp490の親比改善を`exp490-exp357`補正としてexp413へ移植する仮説は棄却し、
weight/component/gate救済、inference、submissionへ進まない。

## 参照ファイル

- `config.yaml`: 入力、anchor、式、gate、実行結果
- `ensemble_contract.yaml`: 凍結したprimary/control契約
- `exp506_..._compact_selfcontained_train.py/.ipynb`: Jupytext実装
- `exp506_..._train.ipynb`: 採用済み正規train Notebook
- `test_exp506_contract.py`: 8件の契約・serialization回帰テスト
- `kaggle/output/stage_a_v2/artifacts/`: version 2生成物
- `SESSION_NOTES.md` / `result.md`: 実行履歴と解釈

## 次

exp506は終端閉鎖する。必要なら低優先のsaved-artifact-only readoutで固定10% report-only controlの
scope / tail寄与だけを説明するが、exp506 gateの再評価や推論候補化には使わない。
