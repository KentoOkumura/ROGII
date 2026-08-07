# 要件

## 依頼

- `GR -> emission -> HMM/PF`の順序は維持したまま、HMM/PFを実行する前に、GR平滑化が
  raw-GR emissionのtruth-nearest候補識別性を改善するかを0-boosterのreadoutで判定する。
- 1件目を`exp304_gr_denoiser_emission_separability_readout`として設計確定する。今回はsteering、
  experiment scaffold、設定、記録だけを作り、平滑化、score計算、HMM、PF、Beamは実装・実行しない。
- `raw`、`robust_rts`、`swt_db4_l3`、`l1_trend`を、exp280と同じ固定shift bank、block、fold、
  raw emission calibrationで比較する。exp189のrolling median / Savitzky-Golayは再試行しない。
- Lateフェーズ専用の評価、分岐、gateは設けない。
- 2件目のtempered raw/smoothed exact-HMM、3件目のRTS uncertainty-aware exact-HMM、
  4件目のHMM通過後PF transfer containment auditをexp304配下の予約契約に固定する。

## 追加依頼（2026-07-20）

- ユーザーの「exp304を実装してください」を、事前登録済みcontractどおりのtrain-side readout実装承認とする。
- 既存の正規Notebook guardは明示採用まで上書きせず、別名compact self-contained Jupytext source / Notebookを作る。
- Kaggle package作成、push、full実行、output取得、HMM/PF/Beam、inference、submissionは今回の承認に含めない。

## 制約

- Route: `pf_beam`。ただしexp304自身はHMM/PF/Beamを走らせないdeterministic diagnosticである。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 評価面はexp226 group-safe OOF `tvt_geop`とexp280の13 shift・512-row block・5 fold identityを固定する。
- score入力はraw train/testに共通するGR、MD、well/typewell、exp226 `tvt_geop`に限定する。
  true TVT、exp226 `tvt_pred` / `gr_delta` / `error`、formation、既存誤差をscore凍結前に読まない。
- denoiserは観測GRだけから一意に決まる固定設定とし、true TVT、fold score、HMM/PF RMSEで
  window、wavelet、level、Q/R、自由度、lambda、betaを調整しない。
- rawを捨てる直接置換はexp304では評価しない。後続の最初のdecodeは必ず
  `(1-beta) * logp_raw + beta * logp_smooth`とする。
- exp209/exp280のknown-prefix sigma、sigma clip `[10, 60]`、log-likelihood clip `600`、
  missing-GR policyを全variantで共有し、smoothed GRからsigmaを再推定しない。
- 学習モデル0、LightGBM config 0、trained fold 0、booster 0、HMM/PF/Beam well-run 0、
  GPU/TPU/internet/inference/submissionなし。Kaggle pushは別承認まで行わない。
- 予約した案2〜4はexp304のvariantではない。各開始条件を満たした後に別expとしてsteeringを作る。

## 受け入れ基準

- steering 3文書、experiment scaffold、`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `reserved_followup_contract.md`、`result.md`、`metrics.json`が実装完了・未実行の状態で整合する。
- rawと3 denoiserの定義、入力面、candidate bank、score freeze、truth attachment、評価scope、
  promotion gate、候補選択tie-break、失敗時の打ち切りが一意に記録されている。
- 案2〜4について、開始条件、依存関係、固定式、比較対象、成功条件、禁止事項、分岐順序が
  `reserved_followup_contract.md`に記録され、他文書から参照されている。
- compact self-contained train source / Notebookに、入力preflight、3 denoiser、solver status、streaming freeze、
  late truth join、scope/fold metrics、technical/quality gate、全expected artifact保存が展開されている。
- synthetic unit test、構文、F821、Jupytext変換test、experiment validationが通る。
- Kaggle実行・HMM/PF再実行・提出が今回の完了条件に混入していない。
- deterministic diagnosticとして実行する将来時点では、input SHA、denoised-GR content SHA、
  target-free score content SHA、scientific contract SHA、Kaggle kernel versionを記録する。
  model/prediction/submissionは生成しないため、それらのSHAは非該当と明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
