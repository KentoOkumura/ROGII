# 要件

## 依頼

`exp335_signed_residual_meta_on_exp264`で使っているlikelihood-PFの
`likpf_mean` primitiveを、`exp404_scale5_sigma_gr_likelihood_pf_ablation`で
保存済みの`likpf_scale_5_x1p0`へ全面置換する実験を設計する。

今回はバックログ、実験ディレクトリ、steeringを作成して設計を固定する。
実装、正規Notebook編集、Kaggle package作成、push、学習、推論、提出は行わない。

## 後続承認

2026-07-26のユーザー依頼`exp413を実装してください`により、train-side helper、
Jupytext形式の別名Notebook候補、専用testの実装が承認された。既存正規Notebookの
採用は未承認のまま、2026-07-27から2026-07-28の段階別依頼によりStage 0、
Stage C、Stage SのKaggle package/push/runだけが順に承認され、完了した。
Stage D version 2は2026-07-28に固定primary gateをPASSした。
2026-07-29のユーザー依頼`推論に進んでください`により、同じexp413内の
current-test推論実装、Kaggle package/push/run、予測監査生成物の取得が承認された。
2026-07-29の後続依頼`submission.csvを生成してください`により、検証済み
predictionをsample submissionのID順・`id,tvt`列へ変換し、提出前検証することが
承認された。さらにユーザー指摘により、ローカルへ変換したCSVではCode
Competitionの提出物にならず、Kaggle Notebook自身のoutputとして
`/kaggle/working/submission.csv`を生成する必要があることを確認した。同じ
current-test full inference NotebookをKaggle version 3として再実行し、その
Kaggle outputを取得・検証する。既存正規Notebookの上書きと外部提出は未承認とする。
その後ユーザーがversion 3をcode submissionしたが、ref `55078306`はhidden
dataset再実行中の未処理例外でscoreなしとなった。raw APIの
`errorDescription`、hidden成功済み親exp335との差分、source監査により、
公開testの14,151 rows / 3 wellsをruntime hard assertしたことを原因と判断した。
同じ科学条件のversion 4では、この公開test固定assertだけをsample submission
由来の動的row / ID / nonempty-well契約へ置換する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親は現行Public-LB reference anchorの`exp335_signed_residual_meta_on_exp264`
  （OOF RMSE `8.146107755881022`、Public LB `7.517`）に固定する。
- PF primitiveの変更は`scale 5`、`gs×1.0`だけとする。`gs×1.3`、scale
  3/8/12、temperature、particle、seed、dynamics、resamplingは変更しない。
- 13候補目を追加しない。12候補のID、順序、family、legal domainを維持し、
  `likpf_mean` slotの値sourceだけを全面置換する。
- `likpf_mean`に直接または推移的に依存する候補、base feature、bank feature、
  selector compact、signed-residual compactをすべて再生成する。旧mean値を
  downstream model inputへ残さない。
- 変更後の候補内容に合わせ、exp264 strict-nested selector 40 CPU boosters、
  exp335 signed selector 20 CPU boosters、downstream LightGBM 15 GPU boostersを
  再学習する設計とする。saved exp335 controlの再学習は0。
- train-side scale-5 predictionはexp404のSHA固定済み生成物を再利用し、新規PFは0。
  current-testではexp072互換のper-well stable seedを維持し、既存trajectoryから
  scale-5出力を保持する。
- CV/LB改善を目的とするlate-stage実験として、by-well p95、worst well、
  悪化well数は必須診断として保存するが、自動停止gateにはしない。
- 同一OOFでscale、multiplier、candidate subset、feature subset、weight、
  thresholdを救済探索しない。

## 受け入れ基準

- `docs/legacy/steering/20260726-exp413-scale5-likpf-full-replacement-on-exp335/`、
  `experiments/exp413_scale5_likpf_full_replacement_on_exp335/`、
  `KAGGLE_DIRECTION.md`、`experiment_summary.md`に同じ設計契約が記録されている。
- 単一変更、変更される5 candidate slot、固定される7 candidate slot、
  base 273 / compact 74 / signed 23 / final 370の再生成契約が明記されている。
- 実行量が1 replacement variant、outer 5 × inner 4の40 CPU selector、
  outer 5 × inner 4の20 CPU signed selector、3 configs × 5 foldsの15 GPU
  downstream、合計75 boosters、control再学習0として固定されている。
- primary gateはsaved exp335比pooled RMSE `>=0.03 ft`改善、3/5 folds以上
  nonworse、near/mid/1000+とhidden-like 2面の各delta `<=+0.02 ft`で固定されている。
- train-side Stage 0/C/S/Dが段階別承認の範囲で完了し、Stage Dは
  saved exp335比`0.261304961 ft`改善、5/5 folds nonworse、全固定scope改善で
  primary gateをPASSしている。
- current-test推論は保存済み40/20/15 modelsをSHA検証してCPU適用し、
  booster学習0、stable per-well seed、scale5 semantic slot全面置換を行う。
  runtimeのrow数、well数、well IDはhidden datasetから動的に決まり、公開testの
  14,151 rows / 3 wellsをhard assertしない。
  Kaggle Notebook内で検証済みpredictionからsample互換
  `/kaggle/working/submission.csv`を生成する。取得後にsampleとの互換性を
  検証し、外部提出はfail-closedとする。
- deterministic anchorとは扱わない。将来実行する場合は、input、feature
  schema/content、candidate/formula、model manifest、OOF prediction、
  current-test prediction、submission、Kaggle kernel versionを記録する。
- gzip生成物を比較する場合は、raw `.csv.gz` SHAではなくdecompressed content
  SHAを主証拠として記録する。

## 次

version 3の公開commit runは完了したが、code submission ref `55078306`はhidden
再実行中の未処理例外でscoreなし。公開test固定row / well assertを除去した
version 4を同じkernel IDで実行し、公開commit outputがversion 3 prediction /
submissionとexact parityであることを確認する。predictionを
sample submissionの14,151 ID順へstrict joinした`id,tvt`だけの
`/kaggle/working/submission.csv`を生成した。取得後のcheckerは行数、列順、
ID順、重複、missing、NaN/inf、source parity、SHAをPASSした。外部提出は
Codexからは行わない。version 2 output取得後にローカルで作ったCSVは事前検証に
のみ使い、Kaggle生成の提出物とは扱わない。
