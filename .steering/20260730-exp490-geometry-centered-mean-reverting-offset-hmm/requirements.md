# 要件

## 依頼

exp357 の geometry-centered residual-offset exact HMM が、GR 尤度で誤った
offset を選んだ後にその誤りを長く維持する問題へ、exp226 geometry へ戻る
平均回帰を追加する実験を設計する。

このターンの範囲は、バックログ登録、steering、実験ディレクトリと設計文書の
作成までとする。HMM 実装、notebook 実装、Kaggle package / push / run、
inference、submission は行わない。

## 制約

- Route は `pf_beam` とする。
- 科学的親は `exp357_exp226_huber_emission_independent_audit` とする。
- 変更する要素は、residual offset と residual offset-rate の遷移平均へ加える
  geometry-centered mean reversion だけとする。
- exp357 の Huber `delta=1.345`、GR 尤度、既知prefixからのsigma推定、
  欠損GR補完、state grid、process noise、posterior mean は固定する。
- 平均回帰の half-life は、対象行が属する exp226 K16 区間の MD span 1区間分
  に固定する。half-life、Huber delta、state grid、process noiseの探索はしない。
- exp226 `tvt_geop` と事前固定した segment / fold / manifest だけを
  candidate生成へ使い、unknown suffix の正解TVT、error、episode roleは
  prediction freeze前に読まない。
- exp357、exp281、exp226 の保存予測をcontrolとして読み、control HMMを
  再実行しない。
- Stage 0は保存済みfixed32による機構確認であり、CVや昇格判定とは呼ばない。
- Stage 0実装・実行とStage 1 full OOF実行は、それぞれ別のユーザー承認を必要とする。
- Stage 1は事前固定した1 variantだけを評価し、同一OOF上のrescueやwinner選択をしない。
- 再現性は `docs/06_reproducibility.md` に従い、入力・契約・prediction・manifestの
  SHAを記録する。gzip生成物の比較はdecompressed content SHAを主証拠とする。
- 実装時もML学習、PF、Beam、GPU、boosterは使わない。

## 受け入れ基準

- 人間が読める状態式、half-lifeの定義、行・区間境界での適用順序が一意である。
- exp357から変更する変数と固定する変数が分離されている。
- Stage 0 fixed32とStage 1 full OOFの実行量、technical gate、scientific gate、
  fail-closed条件が事前登録されている。
- candidateがexp357を改善するだけでなく、exp226 final
  `9.427109596582222`を少なくとも`0.02 ft`上回ることを最終昇格条件に含む。
- by-well tail、1000+、hidden-like 2面、persistent-offset episodeを
  pooled RMSEと同時に判定する。
- 現在の状態がdesign-only / 未実装 / 未実行であることが、実験文書と
  `metrics.json`から誤解なく分かる。
- deterministic anchor として扱わず、将来の実行時にkernel version、
  input SHA、contract SHA、decoder manifest SHA、prediction SHAを記録する。

## 2026-07-30 実装承認後の状態

- ユーザーの明示依頼により、上記design-only turnとは別にStage 0 fixed32の実装を
  承認済みとする。
- compact self-contained train candidate、unit test、leakage sentinel、
  technical / mechanism gateまで実装済みである。
- Kaggle package / push / run、Stage 1、inference、submissionは未承認・未実行のまま
  維持する。

## 2026-07-30 実行承認後の結果

- ユーザーの「実行してください」により、canonical train採用とStage 0 fixed32の
  Kaggle private CPU 1回実行を承認済みとした。
- version 1はCOMPLETE。technical 12/13、mechanism 6/7 PASS。
- full 773 runtime投影`51464.889494 sec > 30600 sec`と、
  matched-control by-well delta RMSE p95
  `+3.118472 ft > +0.25 ft`をFAILした。
- 固定all-AND契約により`stage0_fail_closed`。Stage 1、inference、
  submissionは実装・実行せず、同sample上のrescueも行わない。

## 2026-07-31 full OOF明示承認

- ユーザーの「full wellに進んでください」を、Stage 0のtail/runtime FAILを
  承知したうえで、固定済み1 variantをfull 773 wellsで評価する明示承認とする。
- Stage 0のFAIL判定やgateは書き換えない。今回のStage 1は
  `explicit_user_override_after_stage0_fail`として履歴を分離する。
- 科学条件、half-life、HMM state、Huber emission、保存controlは変更しない。
- 単一kernelの実測投影`51,464.889494 sec`はKaggle上限を超えるため、
  `sha256("exp490::full_well_shard::<well>") mod 4`の4 CPU shardへ分割する。
  shardは実行上の分割だけで、合計candidateは1 variant × 773 HMM well-runsとする。
- 4 shardはtruth / fold / error / episodeを読まずtarget-free predictionとSHAだけを保存する。
  4 shardのprediction SHAを固定した後、別の0-HMM mergeでtruth、保存exp357、
  exp226、hidden-like、persistent episodeを後付けして既存Stage 1 gateを評価する。
- inferenceとsubmissionは引き続き未承認とする。

## 2026-08-01 current-test推論の明示承認

- ユーザーの「念のためLBも見たいので推論に進んでください」を、Stage 1の
  by-well p95 / worst-well FAILを維持したまま、固定済みexp490をcurrent testで
  1回だけ再生成して提出候補を作る明示承認とする。
- 実行根拠は`explicit_user_lb_audit_after_stage1_tail_fail`とし、exp490の
  fail-close判定や物理モデルを再分類・変更しない。
- K16 half-life、Huber、state/rate grid、process noise、initial prior、posterior mean、
  missing-GR処理はfull OOFと完全に同じにする。blend、selector、tail gate、clip、
  fallback、LB向け調整は加えない。
- current testのexp226 geometry-only `tvt_geop`は、既存exp226 inference sourceを
  source/config SHA固定で読み、full-train 773 wellsから3 test wellsへ再生成する。
  exp226のfinal GR correction / U projectionではなく、OOFと同じ`geop`列を使う。
- sample submissionが示す3 wells / 14,151 rowsだけを3 HMM well-runsでdecodeする。
  LightGBM config、trained fold、booster、PF、Beam、GPU、control再実行は0とする。
- notebookは`submission.csv`を生成してよいが、competition submitはこの承認に
  含めない。sample互換、有限値、ID順序、SHA、offline metadataを検証後に別判断する。

## 2026-08-01 competition submit明示承認

- ユーザーの「exp490の推論を提出してください」を、検証済みinference kernel
  version 1のcompetition submit 1件に対する明示承認として記録する。
- submission SHA
  `3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`
  とkernel id `kentookumura/exp490-geometry-mean-revert-offset-hmm-inference`を固定する。
- exp490 Stage 1 fail-closeとtail riskは維持し、LB audit提出と昇格判断を分離する。

## 2026-08-01 competition submit結果

- submission refは`55163886`。Kaggle API statusは`COMPLETE`だが、hidden再実行が
  未処理例外となりPublic scoreは生成されなかった。
- submit-check済み公開CSVではなく、公開sample SHA、14,151 rows、3 wellsを固定する
  inference runtime contractがhiddenデータと非互換である。
- hidden-dynamicなinference version 2の実装と再提出は、この1件の提出承認には含めない。

## 2026-08-02 hidden-dynamic inference修正承認

- ユーザーの「修正してください」を、同じexp490内でhidden再実行に対応する
  inference version 2を実装・静的検証する承認として扱う。
- 公開sampleのSHA、14,151 rows、3 wellsは再現性用の参照値として保持するが、
  runtime pass/fail条件には使わない。
- runtimeのrowsとwell集合は、その実行でmountされた`sample_submission.csv`から導出する。
- sample ID、raw testのunknown-suffix row、exp226 geometry IDの完全一致、sample順序、
  unique ID、finite predictionをhidden-dynamic technical gateとする。
- exp226 source/config SHA、full-train 773 wells、exp490 scientific contract SHA、HMM式、
  K16 half-life、Huber、state/rate grid、process noise、posterior meanは変更しない。
- Kaggle push、実行、再提出はこの実装修正承認に含めず、別承認まで無効にする。

## 2026-08-02 hidden-dynamic inference version 2実行承認

- ユーザーの「実行までしてください」を、canonical kernelのversion 2をprivate CPU /
  internet offでpushし、current public testで完了まで監視する明示承認として記録する。
- 実行量はscientific variant 1、exp226 full-train fit 1、exp226 geometryとexp490 HMMは
  runtime sample well数。model config / trained fold / booster / PF / Beam / GPUは0。
- 親control、Stage 0/1、train shard、mergeは再実行しない。
- 完了後はKaggle outputを取得してsubmission候補を検証してよいが、competition submitは
  この承認に含めない。
