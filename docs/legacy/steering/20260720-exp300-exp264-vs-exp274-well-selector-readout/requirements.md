# 要件

## 仮説

exp264のexp274比悪化はlong-tail深部の一部wellへ集中し、target-free well品質、予測不一致、またはcorrected selectorのcandidate family / margin / switchに再現可能な偏りがある。

## 依頼

- exp264 corrected Stage D v3 OOFを、前のML submitted anchor exp274 raw CatBoost OOFとwell/row単位で比較する。
- exp274より悪化したwell、long-tail上の悪化区間、raw train/typewell特徴を再現可能な生成物として保存する。
- exp264 corrected Stage C v6 selectorについて、候補の選ばれ方、margin、switch、悪化wellでの候補分布を調査する。
- 先行調査で作成した`/tmp/exp264-well-analysis`と一時スクリプトをリポジトリへ移し、移動確認後に`/tmp`側を削除する。
- 追加依頼: 悪化wellで候補familyが多いという相関だけでなく、row-levelでselectorの選択候補が切り替わったこと自体が悪化を生んだかを直接分解する。
- 再訂正: 時系列switchではなく、悪化wellでselectorがactual error上のoracle最良候補を選べず誤候補を選んだことと、Stage D downstream応答のどちらが悪化源かを分解する。

## 制約

- Route: `ml_model`。予測を生成しないOOF診断であり、exp264由来のPF/HMM候補利用はlineageへ記録する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- corrected exp264 Stage D v3 / Stage C v6だけを使用し、feature availability leakageで無効化済みの旧Stage C/D生成物を使わない。
- 新規model fit、candidate regeneration、inference、submissionを行わない。
- exp274はLB submitted anchorだがtrain-side rejectedであり、比較値を新しいCV anchorとして扱わない。
- exp264とexp274でouter foldが異なる63 wellを明示し、cross-experiment OOF診断とmatched ablationを区別する。

## 受け入れ基準

- 773 well / 3,783,989 OOF rowのID、target、prediction coverageを検証する。
- exp274比のwell別悪化一覧、distance/relative-tail/GR/target-residual bucket、SSE寄与を保存する。
- selectorのglobal/悪化群別candidate選択率、candidate lift、dominant candidate、margin、switch、prediction disagreementを保存する。
- selector切替rowと安定row、切替近傍window、候補run単位のSSE差を保存する。新しい候補へ切り替えたrunと、直前候補をそのrun中維持するcounterfactual hard-pathをactual TVTで比較する。
- primary候補について、selector top1とactual absolute error最小のoracle候補をrow単位で比較し、top1一致率、selected-vs-oracle regret、selected hard-vs-exp274、Stage D-vs-selected hardのSSEを保存する。
- `Stage D - exp274 = (selected hard - exp274) + (Stage D - selected hard)`を全体、`>3 ft`悪化well、その他、およびwell別に分解する。oracle候補は診断専用であり、deployable policyとは扱わない。
- selected Self-GR/LikPF・oracle K16の誤ranking pairについて、件数、selected/oracle MAE・RMSE・regretを保存する。Beam誤選択は悪化群とその他でrow件数、全row率、Beam選択内誤選択率、rate lift、母数調整後excess rowsを報告する。
- exp274にはselectorがないため、「exp274から選択候補が変わった」とは定義せず、exp264 Stage C hard top1の時系列切替とexp264 Stage D finalのexp274比悪化を区別する。
- target-free特徴とoracle特徴を分離し、routerへ直接使えない診断を明記する。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`を更新する。
- `/tmp/exp264-well-analysis`、`/tmp/exp264-analysis-exp274`、今回作成した`/tmp/exp264_*analysis*.py`が削除済みである。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 次

診断で見つけたposthoc偏りは直接gateにせず、既存exp276の事前固定済みtarget-free risk監査でouter-fold再現性を確認する。
