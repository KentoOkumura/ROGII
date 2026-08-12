# 要件

## 依頼

- 案3として、exp287の精度が悪いwell側の裾を改善する実験を設計確定する。
- この段階ではバックログ、steering、実験ディレクトリだけを作成し、実装、Kaggle push、学習、推論、提出は行わない。

## 2026-07-21 実装依頼

- ユーザーの「exp334を実装してください」を、設計固定済みtrain候補、fail-closed inference候補、専用test、設定・記録の実装承認として扱う。
- 正規notebook scaffoldは上書きせず、Jupytext percent形式の`*_compact_selfcontained_train.py` / `*_compact_selfcontained_inference.py`と対応notebookを別名で作る。
- Kaggle package/push、preflight実行、15-booster train、inference、submissionは引き続き承認範囲外とする。

## 2026-07-21 実行依頼

- ユーザーの「実行してください」を、compact self-contained train候補の正規train notebook採用、0-booster Kaggle preflight、その成功後の1回の15 GPU booster trainの明示承認として扱う。
- 実行量は1 variant × 3 LightGBM configs × 5 folds = 15 GPU boosters、control再学習0から変更しない。
- inference、submission、control再学習、guard緩和、追加rerunは承認範囲外とする。

## 背景と根拠

- exp287は保存済みOOFでRMSE `8.136708220359452`、exp264比 `-0.3241030172530248 ft`、5/5 folds改善、Public LB `7.530`で現行ML anchorになった。
- 一方、exp264比のworst-well deltaは `+8.228409822385604 ft` で、`+1/+3/+5 ft`悪化well数も `135→140 / 39→40 / 14→19` とすべて増え、train promotion guardには失敗した。
- 仮説は、row単位L2損失ではsuffix評価行が多いwellほど学習損失への寄与が大きくなり、well単位の裾を悪化させる一因になっている、である。これは現時点では未検証の仮定であり、本実験で反証可能な形にする。

## 制約

- Route: `ml_model`
- 親実験および主control: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail guardの比較control: `exp264_exp263_candidate_confidence_dual_selector`
- exp287の421特徴、outer 5-fold group split、target、3 LightGBM config、seed、GPU再現性設定を固定する。
- 変更する変数は、outer-trainの各学習行へ付与するwell均等化sample weightだけとする。
- valid Datasetにはsample weightを付けず、early stopping、fold RMSE、pooled OOF RMSE、scope/by-well評価はすべて従来どおり非加重とする。
- exp287保存済みOOF・fold metrics・by-well metricsをcontrolとして使い、exp287またはexp264のcontrol boosterは再学習しない。
- active variantは1、LightGBM configは3、foldは5、承認後の予定量は `1 × 3 × 5 = 15 GPU boosters`、control再学習は0とする。
- hard-well error、inner-OOF error、target、outer-valid errorを重み作成に使わない。これらを使う再重み付けは別仮説・別実験とする。
- feature追加/削除、formation生成変更、weight式のgrid、loss変更、LightGBM parameter変更、seed bagging、selector変更、後処理、guard緩和を混ぜない。
- `docs/06_reproducibility.md` に従い、GPU学習、SHA記録、Kaggle packageの再現性情報を実装前から固定する。

## 学習重みの契約

- outer foldごとにouter-trainの評価対象行だけを使い、well `w` の行数を `n_w`、outer-train総行数を `N`、well数を `W` とする。
- well `w` の各行の重みを `weight_i = N / (W * n_w)` とする。これにより各wellの総重みは `N / W`、outer-train全体の平均重みは1になる。
- 行数はfoldとscore-row identityからのみ計算し、target、予測、誤差、formation値を参照しない。
- 全3 LightGBM configへ同一の重みを付ける。valid側の重みは常に未設定とする。
- 実装時は各foldでfinite、正値、平均1、well別総重み一致、row identity一致をfail-closedで検証する。

## 受け入れ基準

設計完了は次をすべて満たすこととする。以下は設計確定時点の履歴であり、2026-07-21の実装依頼で実装コード追加だけが上書き承認された。

- steering、実験ディレクトリ、`KAGGLE_DIRECTION.md`、`experiment_summary.md`に同じ仮説・単一変更・実行量・禁止事項が記録されている。
- `config.yaml`で実装、Kaggle push、学習、推論、提出が未承認かつ無効になっている。
- 実装コードは追加されず、notebookは未実装のscaffoldである。
- `make validate-exp EXP=exp334_equal_well_loss_weighting_on_exp287`と実験文書監査が通る。

将来、学習実行が別途承認された場合のpromotion gateは次をすべて満たすこととする。

- 非加重pooled OOF RMSEがexp287比 `+0.02 ft`以内。
- 5 folds中4 folds以上で非加重fold RMSEがexp287以下。
- near / mid / 1000+ / hidden-like各scopeのRMSEがexp287比 `+0.02 ft`以内。
- by-well RMSE deltaのp95がexp287以下。
- exp264比worst-well deltaが `+0.25 ft`以内。
- exp264比 `+1/+3/+5 ft`悪化well数がそれぞれexp264から増えない。
- gate不通過時はguardを緩和せず、本仮説をcloseまたは別仮説として切り出す。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、Kaggle kernel versionを記録する。submissionは本実験の承認範囲外である。
- gzip生成物を比較する場合は、raw `.csv.gz` SHAではなくdecompressed content SHAを主証拠として記録する。

実装完了はさらに次を満たすこととする。

- 保存済みexp287のOOF/model/metrics/fold/by-well/formation manifest/raw schemaと10個のformation fold cacheをSHA固定して再利用する。
- exp287と同じStage C compact 74列とclean base 273列を再構成し、model manifestの421列順と一致させる。
- 5 foldsすべてのouter-train weightをbooster fit前に検証し、weight summary/content SHAを保存する。
- LightGBM fitへの科学的差分は`sample_weight=train_weights`だけで、validation weightを渡さない。
- 専用test、Jupytext round-trip、py_compile、ruff F821、strict experiment validationが通る。
