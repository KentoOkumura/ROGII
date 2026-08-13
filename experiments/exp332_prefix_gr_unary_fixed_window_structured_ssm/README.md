# exp332_prefix_gr_unary_fixed_window_structured_ssm

## 状態

- ルート: ensemble
- 状態: Stage 0 runtime gate FAIL・branch closed
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-21
- 親実験: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`
- 先行条件: exp331はStage A科学gate FAILでbranch close済み

## 仮説

exp295と同じsoft structured objectiveを256-row windowへ限定し、各well・各epochを3 scheduled slots（最大3 active windows）に固定すれば、window内のtransition-aware learningを保ちながらexact DPの計算量を約5%以下へ削減できる。

## 変更点

- exp295のobjective family/weight/sigmaを維持する。
- full suffixを`256 rows × 最大3 non-overlap windows/well/epoch`へ置き換える。確保できないslotはinactiveにする。
- interior teacher boundaryはloss初期化だけに使い、encoder inputやvalid/testへ持ち込まない。
- 評価はwindow分割せずofficial prefixからfull-well exact SSMを実行する。

## 検証方針

- fold/controls/promotion gateはexp331・exp295と同じ。
- Stage 0は固定16 windowsのT4 microbenchmark。8.5時間/14 GBを必須にする。
- Stage Aはfold 0、architecture 1、seed 42、neural model 1だけ。
- window lossの改善ではなくfull-well posterior-mean RMSEとGR attributionで判定する。
- exp331との同時実装・同時GPU比較は禁止する。

## 実行入口

実装時は正規Notebook scaffoldを維持して次のcompact self-contained候補を作成し、Stage 0実行承認後にtrain候補を正規train Notebookへ採用する。

- `exp332_prefix_gr_unary_fixed_window_structured_ssm_compact_selfcontained_train.ipynb`
- `exp332_prefix_gr_unary_fixed_window_structured_ssm_compact_selfcontained_inference.ipynb`

Stage 0完了後は`execution.selected_stage=implementation_only`、`kaggle_push_approved=false`へ戻した。runtime gate FAILのためStage A/B/C、推論、提出は閉鎖している。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |
| Stage 0 保守的fold外挿 | `13.151137275 h`（FAIL） |
| Stage 0 p50 fold外挿 | `12.744535682 h` |
| peak GPU memory | `1.203262806 GB`（PASS） |

## 所見

### 良かった点

- exp295のglobal-structure learningを短区間内では維持する。
- scheduleはtruth/error非参照で全8 epochs分を先にfreezeし、置ける場合は最大3本を確保する。
- interior teacher boundaryは別manifestに分離し、encoderにはofficial `TVT_input`だけを渡す。

### 悪かった点

- window境界をまたぐ長距離driftを学習できず、teacher boundaryで難しさを過小評価する可能性がある。

### リスク / 注意

- 4 DP sweepsは残るため、exp331よりruntimeと実装リスクが高い。
- window長、数、boundary、loss weight/sigmaの救済gridは禁止する。

## 次

事前契約どおりbranch close。window/loss/decoder救済、Stage A/B/C、推論、提出へ進まない。

## 表記

用語は`backlog/KAGGLE_DIRECTION.md`と`docs/glossary.md`に合わせる。
