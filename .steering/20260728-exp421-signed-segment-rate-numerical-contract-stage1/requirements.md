# 要件

## 依頼

exp418 Stage 0で唯一FAILしたmatrix / sequential integration差
`6.295408638834488e-12 ft`を、実用上無視できるfloat64演算順差として扱う。
exp418の判定は変更せず、truth-freeに事前固定した数値契約を持つ後継実験で、
exp418と同じsigned K16 rate Stage 1を実行する。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従う。
- 親exp418の`technical_fail` / `FAIL_CLOSE_BRANCH`記録をPASSへ変更しない。
- 数値許容値は`1.0e-10 ft`に固定する。これはexp418 Stage 1と共通validationに
  既に事前記載されていた値で、OOF scoreから選び直さない。
- exp418 Stage 0 summary file SHA
  `07c719e0f174b1712650563620f6331504dbd1333969c8777f41ce46419dc412`
  を固定する。
- exp418 Stage 0のtechnical FAILが`integration_parity`だけであり、他8 checksと
  pooled / 5-fold scientific threshold checksが成立したことを厳密に検証する。
- truthを読まないfixed synthetic rate vectorでmatrix / sequential integrationの
  cross-runtime差が`1.0e-10 ft`以下であることを学習開始前に検証する。
- signed-rate target、K16 assignment、exp333 nested folds、136特徴、LightGBM
  `lgb1`、sample weight、Stage 1 scientific gateはexp418から変更しない。
- active variant 1、model config 1、outer fold 5、CPU booster 5。
- exp226 fit / control再学習 / PF/HMM/Beam再生成 / GPUは0。
- inferenceとsubmissionは未承認。

## 受け入れ基準

- synthetic numerical auditが`1.0e-10 ft`以下でPASSする。
- exp418 summaryのSHA、唯一のtechnical failure、8/9 technical checks、
  2 scientific threshold checksが固定値と一致する。
- 3,783,989 rows / 773 wells / 12,368 segmentsの5-fold OOFを完走する。
- Stage 1の既存AND gateを変更せず評価し、PASS/FAILをそのまま記録する。
- feature content SHA、5 model SHA、OOF prediction SHA、summary SHA、
  Kaggle kernel versionを記録する。
- deterministic anchorとは扱わず、submissionは生成しない。
- gzip生成物はdecompressed content SHAを主証拠として記録する。
