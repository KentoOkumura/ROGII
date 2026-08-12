# 要件

## 依頼

rate追従遅れを減らす第4案として、acceleration状態を加える
`exp444_acceleration_state_exact_hmm`を設計する。

2026-07-30のユーザー判断により、exp441/exp442の結果を実装前提にする条件付き
候補から、固定済みacceleration-stateを単独で検証する独立仮説へ変更する。
同日の元依頼「exp444を実装してください」を継続し、compact self-contained
Notebook候補、fail-closed inference、専用contract testまで実装する。
実装後の2026-07-30に、別のユーザー指示「実行してください」により正規train
Notebook採用、Kaggle package、Stage 0Aが追加承認された。Stage 0B/1、
inference、submissionは承認されていない。

## 仮説

rate変化の符号を隠れ状態として持続させれば、弱いGR区間でもtrendを毎行
再発見する必要がなくなり、translation lockを減らせる。

Assumption: exp441のfull-support OU単体はStage 0でpersistent lagを改善できなかった
が、rate trendを持続する明示状態を加えることはOU到達性とは別の機構仮説であり、
固定条件のままtarget-free preflightする価値がある。

## 制約

- Route `pf_beam`。
- 構造参照・一要因controlはexp441、root比較対象はexp209。
- exp441 rate kernelとexp209 position/emission/prior/readoutを固定。
- accelerationは3値`[-0.0005,0,+0.0005] rate/MD-ft`。
- transitionはinterior `0.08/0.84/0.08`、境界外向きmassはstayへ加える。
- initial accelerationはzeroに確率1。
- one-factor controlは保存exp441予測、exp209はroot referenceとする。
- Stage 0Aはidentity-only hashで4 wells、Stage 0Bはfixed32 total 32。
- state count/span/transition/prior grid、trigger、reset、re-anchor、selector/blend禁止。
- exp441/exp442の成否は実装・Stage 0Aの先行条件にしない。
- exp441のStage 0 FAILはnegative contextとして固定し、parameter/gate救済に使わない。
- Stage 0Aまでは実行済み。Stage 0B/1、inference、submissionは未承認。

## 受け入れ基準

- accelerationの単位、support、transition、boundary、initial priorが一意である。
- rate/TVT更新順とexp441からの単一差分が明確である。
- exp441 fixed32 prediction SHAをStage 0A/0B前に固定し、controlを再実行しない。
- Stage 1はexp209 root比較を正とし、terminal close済みexp441 full rerunを要求しない。
- Stage 0A runtime、Stage 0B mechanism、Stage 1 gateが固定されている。
- truth-late、SHA、実行量、独立仮説契約が全文書で一致する。

## 次のアクション

Stage 0Aのruntime projection FAILによりterminal closeする。
Stage 0B/1、inference、submission、同一branch内のruntime/state/kernel/
parameter/gate救済へ進まない。
