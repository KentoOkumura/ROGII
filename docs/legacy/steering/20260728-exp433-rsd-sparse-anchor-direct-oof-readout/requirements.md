# exp433 要件

## 依頼

ユーザーの2026-07-28の依頼により、exp426で凍結済みのRSD-binned GR scoreを
実際のexp226 OOF予測へ直接適用して評価する後続実験について、バックログ、
steering、design-only実験scaffoldを作成し、設計を確定する。

この依頼は設計だけを承認する。実装、正規Notebook編集、Kaggle package /
push / run、inference、submissionは承認しない。

## 2026-07-28 実装承認

ユーザーの「exp433を実装してください」を、compact self-contained Jupytext
train候補、対応する未実行Notebook候補、contract tests、config / 文書更新の
承認として扱う。既存の正規train / inference Notebookは上書きせず、
Kaggle package / push / run、inference、submissionは引き続き未承認とする。

## 仮説

exp426でRSD scoreが有効だったblockは全体の25.593939%に留まったが、
absolute datumは毎blockで観測できる必要はない。凍結済みscoreを一切変更せず、
unsupported blockを遷移だけでcarryする固定Viterbi decoderへ入れれば、
疎なGR anchorでもexp226の累積offsetを実OOF上で改善する可能性がある。

## 制約

- Route: `pf_beam`
- 親実験:
  `exp426_rsd_binned_pattern_absolute_reanchor`
- 基準予測:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- exp426 Stage Aのtechnical FAILとterminal-close判断は変更しない。
- exp426 version 1の凍結済みtarget-free score bank、well manifest、
  input manifestだけをscore入力に使う。
- RSD bin幅、block、offset、support、score、rank、top-3を再計算または変更しない。
- primary predictionはexp426で事前登録済みだった固定Viterbi 1個だけとする。
- blockwise top-1は機構診断だけとし、primaryへの差し替えを禁止する。
- score bankとdecoder contractを検証してprediction SHAをfreezeするまで、
  truth、error、oracle offset、persistent episode、hidden-like roleを読まない。
- coverageは必須reportだが、truth readや科学評価を止めるtechnical gateにしない。
- 全3,783,989 OOF rows / 773 wells / 5 foldsで実際のRMSEを評価する。
- model、LightGBM config、trained fold、booster、HMM、PF、Beam、GPUはすべて0。
- 親実験、control、score bankを再生成しない。
- RNGは使わず、well、block、offset、rowのstable sort順を固定する。
- `docs/06_reproducibility.md`に従い、gzip入力はdecompressed content SHAと
  logical content SHAを主証拠にする。

## 受け入れ基準

- exp426のscore bank `101,231 rows = 7,787 blocks × 13 offsets`と、
  記録済みschema / logical / decompressed SHAが一致する。
- exp226 OOFが3,783,989 rows / 773 wells / 5 foldsで、保存基準RMSE
  `9.427109596582213`を絶対誤差`1e-6 ft`以内で再現する。
- primary Viterbi prediction、blockwise diagnostic、support diagnosticsを
  truth join前にfreezeし、独立rerunのlogical prediction SHAが一致する。
- primaryは全OOF rowでexp226比RMSE gain `>=0.10 ft`、改善fold `>=4/5`、
  1000+ gain `>=0.20 ft`を満たす。
- persistent episode SSEを`>=10%`削減し、episode wellsの`>=60%`を改善する。
- 0--50 / 50--100、raw-GR missing、hidden-like 2面のRMSE regressionを
  各`<=0.02 ft`に抑える。
- by-well RMSE delta p95 `<=+0.25 ft`、worst `<=+5.0 ft`を満たす。
- 全gate PASSでもtrain-side PF/Beam routeの候補に留め、inference /
  submissionは別設計・別承認とする。
- FAIL時はdecoder、transition、support、offset、score、clip、blend、
  activationを救済せずterminal closeする。
- deterministic anchorは独立rerun parity、input / prediction SHA、
  Kaggle kernel versionが揃うまでfalseとする。model / submission SHAは非該当。
