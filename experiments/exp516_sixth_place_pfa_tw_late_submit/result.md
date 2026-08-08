# exp516 6位 `pfA × twGR` LATE SUBMIT再現監査 結果

## 結論

final public sourceから切り出したPF単体`pfA × twGR`はKaggleで完走し、late submission ref `55326266`はPublic `10.056` / Private `8.552`だった。技術再現は完了したが、作者報告のstandalone component Public `7.88` / Private `7.78`は再現しなかった。

2026-08-08、ユーザー判断により、報告スコア未再現の結果を保持したまま実験を`completed`とした。

提出前の公開commit runはcurrent visible test 3井・14,151行。Tesla T4 x2でanchor学習`1197.892s`、learned-emission similarity`1.856s`、PF本体`13.483s`だった。PFは`pfA × twGR`のみ、600 particles × 32 seeds、whole-interval ancestral smoothingで実行された。

ユーザーの明示承認後にKaggle version 1をpushしたが、CRLF raw-file SHAとLF embedded-text SHAを混同したidentity guardが予測開始前にfail-closeした。数値設定を変えず2種類のSHAを分離し、version 2候補は契約テスト`7 passed`と全静的検証を通過した。version 1はPF予測未実行のため科学的negative resultではない。

最終静的検証はcontract test `7 passed`。公開commit runのKaggle生成`submission.csv`は14,151行、ID順完全一致、重複/欠損/非有限値なしで、SHAは`feee82f8...80c`。Notebookとsubmission messageには`LATE SUBMIT`を明示した。

## Late submission結果

- Kernel: `kentookumura/exp516-sixth-pfa-late-submit-inference`, version 2
- Submission ref: `55326266`
- Status: `COMPLETE`（scoring 28分）
- Public / Private: `10.056 / 8.552`
- 作者報告standaloneとの差: Public `+2.176`、Private `+0.772` RMSE（小さい方が良い）
- Public commit output submission SHA256: `feee82f8ec8d24390fe0478a983fef42054c127487ac793612ead3ab61fc080c`
- Hidden rerun: 数値スコア付き`COMPLETE`。hidden well数、工程別runtime、scored output SHAはAPIから取得不可

## 評価契約

公開sourceの単体`pfA × twGR`を固定再現し、technical gate通過後に1回だけlate submitした。作者報告の単体component CV 7.8 / Public 7.88 / Private 7.78は外部参照値であり、exp516の実績ではない。

6位最終systemのCV 5.4577 / Public 5.626 / Private 5.984は、91候補、candidate-curve NN、TCN、GBM、de-shrinkを含む別契約であり、この実験は再現を主張しない。

この提出はコンペ終了後の`LATE SUBMIT`であり、正式順位や競技中のmodel selectionとは分けて記録する。

## 解釈とnegative result scope

閉じる範囲は`(GR-free anchor + final-v96 pfA learned emission, twGR representation, particle TVT path, standalone direct smoothed-mean decode, no fusion, late hidden test, 600×32, T4x2)`である。この固定componentは作者報告standalone scoreを再現せず、現在のPF/Beam route基準にもならない。

ただし、writeupの`twGR-prior PF alone`はstage 2-4の値であり、stage 2-2はfixed-lag 192として説明される一方、exp516が忠実に切り出したfinal v96 sourceはwhole-interval smoothingとlearned emissionを含む。両者が同一artifact・同一decode・同一runtime RNGである証拠はないため、今回の差を6位PF family全体の否定や実装バグと断定しない。

91候補、他representation、candidate-curve NN、TCN、GBM、de-shrinkを含む6位最終systemは別契約で未評価。one-shot late-submit契約を消費したため、LB後のparameter/seed/postprocess調整や再提出は行わない。必要なら次は提出なしでstage 2-4とfinal v96のsource-context差だけを監査する。
