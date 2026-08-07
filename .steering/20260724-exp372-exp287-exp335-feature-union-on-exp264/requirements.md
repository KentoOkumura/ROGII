# 要件

## 依頼

exp287とexp335を統合する候補をバックログへ追加し、新しい実験ディレクトリとsteeringを
作成して設計を確定する。事前提案の0-booster相補性診断は必須化せず、444特徴の直接統合
ルートから始める。実装はまだ行わない。

## 制約

- 対象実験: `exp372_exp287_exp335_feature_union_on_exp264`
- Route: `ml_model`
- 親: corrected `exp264_exp263_candidate_confidence_dual_selector`
- 統合元:
  - `exp287_fold_safe_formation_74_addonly_on_exp264`
  - `exp335_signed_residual_meta_on_exp264`
- 特徴面は`clean273 + saved74 + formation74 + signed23 = 444`の1 variantに固定する。
- exp264 control、exp287/exp335単独親、parent/signed selectorを再学習しない。
- exp287 formation train cacheとexp335 signed compactを保存済みmanifest/partition SHAで再利用し、
  train featureを再生成しない。
- corrected exp264のfold、target、3 LightGBM config、seed、early stoppingを変更しない。
- 実装、正規Notebook採用、Kaggle/Colab package、push/run、推論、提出は本依頼の範囲外。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- `KAGGLE_DIRECTION.md`の未着手バックログに、優先度、依存、検証方法、禁止事項がある。
- `.steering/20260724-exp372-exp287-exp335-feature-union-on-exp264/`に
  requirements/design/tasklistがある。
- `experiments/exp372_exp287_exp335_feature_union_on_exp264/`にdesign-only scaffoldがある。
- `config.yaml`にroute、lineage、入力SHA、444特徴順序、validation、gate、再現性、
  `1 variant / 3 configs / 5 folds / 15 GPU boosters / control 0`が固定されている。
- technical、incremental utility、tail promotionを分離し、FAIL後の扱いが固定されている。
- placeholder以外のNotebookロジック、helper、test、Kaggle packageが作成されていない。
- 実装・train・inference・submissionの承認flagがすべてfalseである。

## 2026-07-24 実装承認追記

ユーザーの「exp372を実装してください」を、compact self-contained train候補と
保存surface union pipeline、専用contract testの実装承認として記録する。

- 実装承認だけを`true`にし、正規train Notebook採用、Kaggle package、push/run、
  inference、submissionは引き続き`false`とする。
- 既存の正規`*_train.ipynb` placeholderは上書きせず、
  `*_compact_selfcontained_train.py` / `.ipynb`を別名で作る。
- 444特徴の組み立て、manifest/partition SHA、formation logical SHA、fold/role alignment、
  15 model slot、固定AND gateを実装・静的検証する。
- train、inference、submissionは実行しない。

## 2026-07-25 推論override追記

ユーザーの「推論に進んでください。」を、固定科学gate FAIL後の別明示overrideとして記録する。

- 同じ`exp372_exp287_exp335_feature_union_on_exp264`内で推論を実装・実行する。
- 保存済み40 parent selector、20 signed selector、15 union TVT modelをSHA検証して使い、
  model fit / booster trainingは0とする。
- raw testから12 candidate、clean273、outer別saved74、outer別signed23、
  all-train-reference formation74を同一CPU runで再生成する。
- 最終特徴順はtrainと同じ
  `clean273 -> saved74 -> formation74 -> signed23 = 444`に固定する。
- Kaggle private CPU Notebookで実行し、`submission.csv`を提出形式検証用に生成する。
- 外部competition submit、再学習、同一OOF救済、gate再分類、Colab fallbackは承認範囲外。
- trainのincremental/tail/promotion FAILは維持する。

受け入れ基準:

- Jupytext percent形式のcompact self-contained inference候補を先に作り、
  既存正規inference placeholderは採用判断まで上書きしない。
- Notebook上で承認、入力/model SHA、raw-test再生成、444列組み立て、15 model予測、
  submission/metrics/SHA保存を追える。
- 40 + 20 + 15 model fileを全件SHA検証し、0 fitを記録する。
- sample submissionのrow/order/header/finite contractをNotebook内で確認する。
- static validation後に同一canonical inference IDへCPUでpushし、完了まで監視する。
- outputを取得した場合は`kaggle-submit-check`で検証するが、competition submit APIは呼ばない。
