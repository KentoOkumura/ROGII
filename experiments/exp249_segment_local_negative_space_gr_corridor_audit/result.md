# exp249_segment_local_negative_space_gr_corridor_audit 結果

## 状態

Kaggle Stage 0 Version 1は表示parity pass。Stage 1 Version 2は773 wells / 3,783,989 rowsを監査してCOMPLETEしたが、7 guards中6 guardsが不通過となり不採用。

## 仮説

局所segment内だけのGR mismatch ridge crossing / corridor transitionなら、exp246のfull-tail hard-history barrierより低い誤警報でbad candidate riskを濃縮できる可能性がある。

## 設定

- Route: `pf_beam`
- 親: `exp246_negative_space_gr_barrier_audit`
- window: 128 rows × 64 bins、±192 ft、stride 64
- active mode: `stage1_full_audit`
- model/config/fold/booster: `0 / 0 / 0 / 0`
- candidate変更、推論、提出: なし

## 結果

Stage 0は3 wells × 3 positions、計9画像を生成した。全画像で128×64 surface、±192 ft crop、TVT右向き・horizontal row下向き、signed `[-4, 4]`、absolute `[0, 4]`、segment/typewell crop別median-IQRを確認した。last windowの末尾行反復はexp202/208と同じclip契約であり、parity passと判断した。

Stage 1のprimary bad-candidate precision liftは0.917349で、最低条件1.5を下回った。good-candidate false-alertは0.540627、truth instantaneous false-alertは0.536992で、許容値0.02 / 0.001を大幅に超えた。overlap disagreementは0.138378、hidden-like truth false-alertは0.523356、worst-well truth false-alertは0.813880だった。boundary側の誤警報がcoreより0.129466低いguardだけはpassしたが、signal全体の誤警報と逆濃縮を救えない。

candidate family別precision liftも`beam_mean` 0.984600、`hyb` 0.973831、`likpf_mean` 0.986695、`pf_ancc` 0.989304、`sc_ens` 0.944813で、すべて1未満だった。1000+ aggregateも0.926499で、長距離tailに限定しても改善しない。

## 実装検証

- `py_compile`、Ruff、Jupytext変換/test、strict experiment validation: pass
- synthetic core contracts: pass
- synthetic well readout: 2,238 view rows / 24 segment summaries
- Kaggle Stage 0 package: Version 1 COMPLETE、生成物取得済み

## Kaggle Stage 0

- kernel: `kentookumura/exp249-segment-local-gr-corridor-audit-train`
- Version 1: COMPLETE
- runtime: 185.900秒
- preview: 9 images / 3 wells
- decision: Stage 0 parity pass、Stage 1有効化

## Kaggle Stage 1

- Version 2: COMPLETE
- runtime: 3,673.976秒（61.2分）
- processed: 773 wells / 3,783,989 rows
- decision: `segment_local_audit_guard_failed`
- guards: 1 pass / 6 fail
- downstream confidence feature化、候補変更、推論、提出: 禁止

## 結論

元仕様の局所segment化は、exp246のglobal hard-history問題を解消しなかった。誤警報の主因はsegment境界ではなく、局所component transition自体がtruthとgood candidateにも高頻度で発生することにある。threshold・segment長・strideの追加探索は行わず、この実装は診断結果として終了する。
