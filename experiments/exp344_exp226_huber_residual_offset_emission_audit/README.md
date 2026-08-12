# exp344 exp226 Huber residual-offset emission audit

## 状態

- Route: `pf_beam`
- 状態: exp342依存pattern不成立、未実装・未実行で閉鎖
- 優先度: P4
- 親実験: `exp281_exp226_residual_offset_exact_hmm_transition_probe`

## 仮説

Student-tが極端残差では効くが全域で尤度を平坦化しすぎる場合、固定delta=1.345のHuberならrobust化範囲を限定してshift識別力を残せる可能性がある。

## 検証方針

- exp342が「極端残差改善あり、全体rankはflatteningで失敗」という事前指定patternになった場合だけ実施可能とする。
- Huberは `delta=1.345`、scaleはexp281 sigma、emissionは負のHuber lossに固定する。
- Stage 0はexp342と同じ512-row block・13 shift・Gaussian control・gateを使う。
- Stage 1はStage 0全通過と別途承認後のみ、新規variant 1個、773 well HMM runとする。
- delta grid、likelihood cap、Student-tとの同時実行、欠損・ACF補正は禁止する。

## 所見

HuberはStudent-tの代替を結果後に自由選択する案ではなく、事前登録したflattening失敗patternにだけ対応する保険枝とする。
exp342は極端残差scopeでは改善したが、必須のStudent-t margin flattening signalが
falseだった。したがって依存patternは不成立で、Huberを実装・実行しない。

## 実装境界

notebookはscaffold placeholderのまま保持する。Kaggle package/push/run、
Stage 1、inference、submissionは行わず閉じる。

## 文書

- Steering: `../../docs/legacy/steering/20260722-exp344-exp226-huber-residual-offset-emission-audit/`
- 設定: `config.yaml`
