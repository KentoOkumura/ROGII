# exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264 結果

## 結論

候補別RMSEを安全に活用する方法を、保存OOF診断上で確立した。

- exp407の悪化原因は、候補別RMSEそのものではなく、逆RMSEを共有モデルの
  sample weightへ変換したことで生じた行単位のscore surface・順位の崩れだった。
- exp415では候補別RMSEを重みにせず、fold-safeな候補方向priorとしてだけ使う。
- 親予測からprior候補方向へ動かす量を各行`±0.25 ft`へ制限する。
- Kaggle private CPU version 1でtechnical `15/15`、scientific `6/6`をPASSした。
- 保存OOF RMSEは`8.5870043867 -> 8.5634739318`、`0.0235304549 ft`改善した。

これは保存OOF診断上の方法確立であり、current-test一般化、LB改善、
route anchor更新、submissionを意味しない。

## 固定policy

```text
parent_pos = argmin(parent_pred_abs_error)
prior_pos  = argmin(parent_pred_abs_error + fit_candidate_rmse)
raw_nudge  = 0.5 * (prior_tvt - parent_tvt)
correction = clip(raw_nudge, -0.25, +0.25)
prediction = parent_tvt + correction
```

候補別RMSEは候補の方向決定だけに使い、sample weight、特徴量、予測量にはしない。
係数1、blend 0.5、cap 0.25は固定し、grid探索や結果後の救済変更はしていない。

## Kaggle確認結果

- Kernel:
  `kentookumura/exp415-fold-safe-rmse-prior-bounded-nudge-train`
- Version / id_no / status: `1` / `128717911` / `COMPLETE`
- Notebook計測時間: `126.338 sec`
- 実行量: variant 1、model / booster / GPU / control再学習 /
  PF / HMM / Beam / inference / submissionはすべて0

| Scope | 親RMSEまたは判定 | exp415または差分 |
| --- | ---: | ---: |
| overall | 8.5870043867 | 8.5634739318（-0.0235304549） |
| fold | - | 5 / 5 改善 |
| 距離bucket | - | 4 / 4 改善 |
| hidden-like | - | 2 / 2 改善 |
| well | - | 544 / 773 nonworse |
| worst well `cc08aa63` | 8.9073208850 | 9.0786999725（+0.1713790874） |

fold別RMSE差は`[-0.024591, -0.031692, -0.015476, -0.026021, -0.020061]`。
距離bucket差はnearから順に`-0.038127, -0.039281, -0.036257,
-0.023321 ft`、hidden-like差はspatial `-0.024995`、
typewell-purged `-0.023287 ft`だった。

3,783,989行中2,685,663行を補正し、2,012,310行がcapに達した。
correction RMSは`0.189787 ft`、最大絶対値は`0.25 ft`だった。

## 根本原因の根拠

- 親hard RMSE: `8.5870043867`
- exp407 hard RMSE: `8.6681410246`
- candidate×fold平均score shiftだけを親へ適用:
  `8.5804769147`、4/5 folds nonworse
- exp407からcandidate×fold平均shiftを除いたrow-local変化:
  `8.6735992633`、1/5 folds nonworse
- final weightとrow-local score差stdのSpearman: `-0.593387`
- final weightとscore MAE差のSpearman: `-0.411670`
- final weightとbinary logloss差のSpearman: `-0.603779`
- final weightとcandidate定数shiftのSpearman: `-0.073243`

したがって主因は単純なcandidate biasではない。inverse-RMSE weightingが
共有木の局所gradient / splitを変え、低重み候補が局所的に有用な行まで弱めた。
候補RMSEをbinary objectiveにも同じ重みで使った点も目的不整合だった。

## Risk certificate

scope内の親誤差を`e`、補正を`d`とすると、Minkowski inequalityにより

```text
RMSE(e + d) - RMSE(e) <= RMS(d) <= max(abs(d)) <= 0.25 ft
```

が任意のwell / fold / bucketで成り立つ。Kaggle上では785 scopesすべてで
3つの不等式を確認し、観測worst-scope悪化は`+0.171379 ft`だった。

## Leakage・再現性監査

- truth-free phaseでfreeze SHAを確定するまでtruth readは0。
- evaluationはfreeze SHA確定後に開始。
- truth reconstruction誤差とcorrection適用誤差はともに0。
- technical gate `15/15`、scientific gate `6/6`。
- ダウンロードした小容量artifactはgate manifest記載SHAと全件一致。
- 実行package SHA:
  `2ef5f85dea2c8b538fa981fc72641adbce69806d8b41bb2246f2c25667832539`
- 実行config SHA:
  `21450bcff8569dd4a2ed84c872af13a01aed488634f61466a130d8c5d22bca90`
- Jupytext source SHA:
  `8494936082f72c1e7e959b87b61890961acb7cbcfa780d95bf10e8b103c13326`
- truth-free freeze SHA:
  `0dd3f55991969da433d65391d5f94efaeff61a615608635767962866a0971aec`
- prediction SHA:
  `cb820ae7c499db8cc6aad37d5665b08e517c88d503a5176b27b03c1b45035f61`
- all-scope metrics SHA:
  `469e3985a4217e0621eb3d5386395aad0b262f7386909fbad442362f723177a8`
- gate SHA:
  `30bc689e5e4fe178735c150adac1879226541ee68560aeb5bfb5bff51d8054a0`
- risk certificate SHA:
  `25ed51b33bfb2123316075b655e43bc832682448abb152de771f8570fbc148ad`

## 実装上の補足

push前のpreflightで、repo rootの別実験用`config.yaml`を誤って先に探索する
path優先順位バグを検出した。exp415実験ディレクトリ内を先に探索するよう修正し、
専用8 tests、関連40 tests、Jupytext round-trip、py_compile、Ruff、
strict validationを再度PASSしてから最終packageを作成した。

初回の57文字slugはKaggle `SaveKernel 400`で実行前に拒否され、kernel/versionは
作成されなかった。科学条件を変えず47文字のcanonical slugへ短縮し、
version 1を完了した。

## 次

exp415は保存OOF診断として完了する。推論や提出は行わない。
current-testへ適用する場合は、未知行に対するfold-independent RMSE priorの
生成契約と、別の一般化検証を新しい実験として事前設計する。
