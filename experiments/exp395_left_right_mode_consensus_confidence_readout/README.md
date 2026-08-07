# exp395_left_right_mode_consensus_confidence_readout

## 状態

- ルート: PF/Beam
- 状態: exp391 Stage A1 FAILにより未実装のまま閉鎖
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-25
- 親実験: `exp391_prefix_anchored_mode_persistence_hmm_readout`

## 仮説

同じstable mode lineageをheel側とtoe側の重ならないGRで別々に評価したとき、
左右posteriorの不一致は、物理モデルが誤った地層対応modeへ入った区間を
truthなしで識別できる。

## 変更点

- exp209/exp391のHMM grammarとmode identityは変更しない。
- event中心の左右に512-row windowを置き、各側64 rowsのgapを空ける。
- `sum_m min(P_L(m), P_R(m))`を唯一のprimary confidenceとする。
- exp226 geometry / LikPF / HMM / exp226 final / exp263のmode一致は
  二次readoutに限定する。
- TVT候補、補正、mode切替、blend、selectorを作らない。

## 検証方針

- Fold: exp391/exp226と同じouter 5 reporting folds
- Group: `well_id`
- Primary scope: exp391 Stage A0でtruth-freeに固定した1,234 events
- Secondary scope: unknown suffixの256-row stride checkpoint
- Leakage Check: checkpoint / mode / confidence / circular nullのSHA freeze前に
  suffix truth / error / hidden-like roleを読まない
- Primary gate: bad10 AUC、fold再現性、confidence quartile RMSE差、
  mode不一致error enrichment、circular null差

## 実行入口

- 学習 notebook: template placeholderのまま。実装しない。
- 推論 notebook: template placeholderのまま。実験はtrain-side診断専用。
- Kaggle package / push / Stage 0 run: 未実行。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- reverse predictorの境界条件差を混ぜず、同じmodeへの左右evidence一致を直接測る。
- 過去のfuture-evidence hard switch失敗を踏まえ、confidence-onlyに限定した。
- exp386/387の空scenario-bank依存を除いた。

### 悪かった点

- exp391 Stage A1はHMM支持1/19 events・1/5 foldsでFAILし、
  HMM mode carrierの適格性を満たさなかった。
- 左右はcalibrationとphysics priorを共有するため、完全に独立ではない。
- 同じ反復GR aliasへ左右とも誤一致する可能性がある。

### リスク / 注意

- exp391 Stage A1 FAILの固定条件に従い、未実装で閉じた。
- historicalなStage 0承認は使用しない。
- confidence gate PASSはRMSE改善やPublic LB 6.5を保証しない。

## 次

実装、Stage 0、full OOF、inference、submissionへ進まない。
