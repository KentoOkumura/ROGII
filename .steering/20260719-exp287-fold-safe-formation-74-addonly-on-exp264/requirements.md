# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `fold_safe_formation_74_addonly_on_exp264` を、
`exp287_fold_safe_formation_74_addonly_on_exp264` として実装する。

## 制約

- Route: `ml_model`。修正版exp264のPF/Beam/HMM由来候補はnested selectorのcompact meta featureとして
  補助利用し、direct blendやhard-pathを行わず、最終予測はdownstream LightGBMが生成する。
- 親は修正版 `exp264_exp263_candidate_confidence_dual_selector` Stage C v6 / Stage D v3だけを使う。
- 追加対象はexp264 availability auditの `status=fail`、`family=base_replay`、
  `dependency=full_train_formation_reference` に一致する74列に固定する。
- exp111系27列、依存GRWR 6列、旧380列surface、full-train formation OOFを使わない。
- outer foldごとにFormationPlaneKNN / DenseANCCImputerをouter-train wellsだけでfitする。
  outer-train targetは自身をreference queryから除外し、outer-validはouter-train referenceだけを使う。
- current-test契約は全train wellsをreferenceにし、target horizontalからformation列を読まない。
- 欠損・nonfinite、train/current-test schema、既存347列とのexact duplicate / Pearson / Spearmanを
  LightGBM fit前に監査し、相関による事後pruneは行わない。
- active variant 1、LightGBM config 3、fold 5、合計15 GPU boosters。保存済みexp264 347列OOFを
  controlにし、control再学習は0。
- Kaggle train prepare/pushは、exp276 corrected revalidationを先行確認したうえで、15 boosterの
  明示承認後だけ行う。exp276 PASS自体は実装・学習のhard prerequisiteにしない。
- guard PASS前のcurrent-test feature生成、model inference、submissionを原則禁止する。2026-07-20の
  ユーザー明示指示は保存済みmodelによるinferenceだけのoverrideであり、guard FAILは保持し、
  再学習、guard緩和、competition submitへ拡張しない。

## 受け入れ基準

- Jupytext percent形式のtrain / disabled inference sourceを作成し、正規`.ipynb`へ変換する。
- feature contractが監査CSVの固定74列・SHA・列順を拒否可能な形で検証する。
- 全5 foldのtrain/valid feature cacheをモデルfit前に生成し、reference / target well SHA、schema SHA、
  logical content SHA、Parquet SHA、duplicate/correlation監査を保存する。
- final feature surfaceがclean 273 + nested compact 74 + formation 74 = 421列である。
- promotion guardをpooled delta `<= -0.02`、4/5 folds改善、near / mid / 1000+ / hidden-like
  delta `<= +0.02`、worst-well `<= +0.25`、+1/+3/+5 ft悪化well数非増加に固定する。
- model manifest、OOF prediction、feature cache、入力、Kaggle kernel versionのSHA記録口を持つ。
- 構文、F821/F401/E9、専用tests、Jupytext test、strict experiment validationが通る。
- override inferenceはraw testからcandidate / clean 273 / compact 74を再生成し、全train wells参照・
  target formation列非読取でformation 74を再生成する。40 selector + 15 TVT modelのSHAを検証し、
  booster学習0、submission file生成可、competition submit無効をfail-closedにする。
