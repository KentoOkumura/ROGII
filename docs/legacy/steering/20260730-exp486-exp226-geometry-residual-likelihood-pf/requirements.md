# 要件

## 依頼

exp279のexp226 geometry-centered unaryとexp281のresidual-offset stateを、
現行temperature-5 likelihood-PFで検証可能な二つの事前固定variantとして設計する。
同一実験内で報告するが、OOF結果から勝者を選ぶ運用は行わない。

2026-07-30の追加依頼`exp486を実装してください`により、compact
self-contained Stage 0候補、fail-closed inference guard、contract testまでを
追加承認する。正規Notebook採用、Kaggle package / push / run、Stage 1、
raw-test inference、submissionは引き続き未承認とする。

同日の追加依頼`実行してください`により、compact候補の正規train Notebook採用、
canonical Kaggle package / pushとfixed32 Stage 0実行までを追加承認する。
Stage 1、raw-test inference、submissionは引き続き未承認とする。

## 根拠

- exp279 geometry unary HMMはexp209を`1.902300 ft`改善したが、exp226より
  `1.797655 ft`悪く、tail/worstが崩れた。
- exp281 residual-offset HMMはexp279を`0.208567 ft`改善したが、exp226より
  `0.400310 ft`悪く、p95 `+10.982960 ft`、worst `+30.961675 ft`だった。
- exp419はexp226のrelative geometry rateだけをdefensive proposalへ使い、
  absolute `tvt_geop` unaryやresidual-offset stateは使っていない。

## 制約

- Routeは`pf_beam`。親exp417、実装親・保存control exp404。
- Variant Aはexp279の`sigma=20 ft, lambda=0.50` absolute geometry unary。
- Variant Bはexp281由来の`TVT=tvt_geop+offset` slow residual-offset state。
- exp226 OOFからprediction前に読める列は
  `well_id,row_idx,suffix_offset,fold,tvt_geop`のみ。foldはfreeze後のreport専用。
- `tvt_pred,gr_delta,tvt_true,error,abs_error`はPF入力禁止。
- 2 variantを独立判定し、same-OOF winner selection、blend、gateを行わない。
- Stage 0は2×fixed32、Stage 1は全PASS・別承認時だけ2×773 wells。
- compact self-contained Stage 0実装とcontract testは承認済み。
- 正規train Notebook採用、Kaggle package / push、fixed32 Stage 0実行は
  承認済み。
- Stage 1、raw-test inference、submissionは未承認。

## 完了判定

2026-07-30にKaggle private CPU version 1でfixed32 Stage 0を完了した。
事前固定runtime投影とstrict residual support boundをFAILしたため
`stage0_fail_closed`とし、Stage 1、raw-test inference、submissionへ進まない。
fixed32 RMSEは記述値でありCVやvariant選択には使わない。

## Stage 1再開承認

同日の追加依頼`実行時間は許容するのでStage 1に進んでください`により、
元のruntime FAILを監査履歴として保持したまま、全773 wellsのStage 1実装・
Kaggle実行を承認する。strict support boundのFAILは最大約`1.1e-15`の
浮動小数overshootであり、`1e-12`以内のnumerical toleranceをtechnical
readbackにだけ適用する。元のStage 0 gateをPASSへ変更しない。

Stage 1は二variantを両方実行し、各variantを保存exp404 controlへ独立判定する。
fixed32記述値から片方だけを選ばず、same-OOF winner selectionも行わない。
raw-test inferenceとsubmissionは引き続き未承認とする。

version 2が両variantのtarget-free freeze後に保存HMMの期待SHA typoで停止した場合、
freeze済みartifactを全SHA検証してprivate Datasetへ固定し、同じ科学contractの
truth-late readoutをcurrent PF rerun 0で再開することを許可範囲に含める。

## 受け入れ基準

- absolute unary式とresidual stateの初期化・遷移・出力式が一意である。
- exp419との非重複、exp226列allowlist、train OOF/test regeneration契約が明確である。
- variant別の実行量、gate、target-free mechanism readout、truth-late、SHAを固定する。
- 二候補間のOOF選択、geometry weight/scale/noise/grid、fallback救済を禁止する。
- compact self-contained train候補とfail-closed inference guardが存在し、
  既存の正規placeholder Notebookを上書きしない。
