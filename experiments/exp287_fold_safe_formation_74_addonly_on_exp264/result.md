# exp287_fold_safe_formation_74_addonly_on_exp264 結果

## 仮説

full-train formation reference依存で除外された74列をouter fold内で再生成すれば、修正版exp264の
clean 273 + nested compact 74へhidden-safeなformation signalをadd-onlyできるかを検証する。

## 設定

- 親: corrected exp264 Stage C v6 / Stage D v3
- Route: `ml_model`
- variant: clean 273 + nested compact 74 + fold-safe formation 74 = 421列
- control: saved exp264 corrected 347-feature OOF
- 学習量: 1 variant × 3 LightGBM configs × 5 folds = 15 GPU boosters
- control再学習: 0
- seed: 42、formation生成はRNGなし
- kernel: `kentookumura/exp287-foldsafe-form74-addonly-exp264-train` version 5

## 結果

Kaggle T4 version 5は15/15 boostersを`25282.477 sec`（約7時間1分22秒）で完走した。

| 指標 | 親exp264 | exp287 | delta（new - parent） |
| --- | ---: | ---: | ---: |
| pooled RMSE | 8.460811 | 8.136708 | -0.324103 |
| fold 0 | 8.468093 | 8.070368 | -0.397725 |
| fold 1 | 8.309700 | 8.255838 | -0.053862 |
| fold 2 | 8.249391 | 7.893630 | -0.355761 |
| fold 3 | 8.450573 | 8.106567 | -0.344006 |
| fold 4 | 8.814966 | 8.349626 | -0.465340 |

5/5 foldsが改善した。距離bucketとhidden-likeもすべて改善した。

| scope | 親exp264 | exp287 | delta |
| --- | ---: | ---: | ---: |
| near 0-250 | 1.583151 | 1.547650 | -0.035501 |
| mid 250-1000 | 4.099686 | 4.047816 | -0.051869 |
| 1000+ | 9.302283 | 8.936648 | -0.365635 |
| hidden-like spatial | 9.420315 | 8.799768 | -0.620547 |
| hidden-like typewell-purged | 9.341391 | 8.735404 | -0.605988 |

## Promotion guard

総合判定は`FAIL`。

- PASS: pooled delta、改善fold数、near / mid / 1000+、hidden-like 2面
- FAIL: worst-well delta `+8.228410 ft > +0.25 ft`
- worst well: `fb03ae90`、親RMSE `29.631678`、exp287 RMSE `37.860088`
- FAIL: 親とclean controlの比較に対する悪化well数の非増加
  - +1 ft: `135 -> 140`
  - +3 ft: `39 -> 40`
  - +5 ft: `14 -> 19`

globalには明確な改善だが、well-level tail safetyを満たさない。事前固定guardを緩和せず、
train完了時点ではcurrent-test生成、inference、submissionを行わない判断とした。
後段の明示overrideによるinferenceとユーザー提出は別判断として記録する。

## Formation feature readout

formation 74列のmean gain上位は`tvt_dense50_d`、`tvt_densew_d`、`tvt_dense_d`、
`dense_nb_std`、`form_mean_d`だった。relationship auditは740行、exact duplicate 0、pruned 0。
相関監査は設計どおりreport-onlyで、結果に合わせた列削除は行っていない。

## 再現性

- rows / wells / models: `3,783,989 / 773 / 15`
- OOF SHA256: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- model manifest SHA256: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- metrics SHA256: `435434342494aaa62cee6e627809363ac34f16174973f4b81301d2923f780862`
- fold metrics SHA256: `864eca0452eea578c96baa653d25c4f2ae241c84b8e5d659b277407b5e427141`
- by-well SHA256: `3562cec13abe3c3df496e57d71b46aeb592ea2022c7bf0b9b5df1e062c21024d`
- formation fold manifest SHA256: `25611e281299991d626f1caca48673aee6225a890ad47ecdcd28a117ae827772`
- raw schema audit SHA256: `45d0bf77b1893adfce74921f4427c4ca5ba6d95c69326cbbd35abb766e502a41`
- train run submission generated: false
- GPU rerun parity: 未確認。単一versionのartifact SHAを記録し、deterministic rerunを主張しない。

## 失敗履歴

- version 1: 12.03秒、input mount path resolver、booster 0
- version 2: 475.305秒、corrected parent OOF SHA pin、booster 0
- version 3: 516.087秒、duplicate `last_known_tvt` projection、booster 0
- version 4: 533.925秒、formation reference availability契約、booster 0
- version 5: COMPLETE、15/15 boosters、scientific guard FAIL

## 結論

fold-safe formation 74列のglobal signalは確認したが、固定worst-well / 悪化well数guardを満たさないため
train-side判定は`train_complete_guard_failed`のままにする。同一OOFでのfeature/grid/threshold救済と
guard緩和は行わない。

## 2026-07-20 inference-only override

ユーザー明示指示により、failed guardをPASSへ変更せず保存済みmodel inferenceだけをoverrideした。
raw testからexp263 12候補、clean 273、outer別compact 74を再生成し、全773 train wellsだけをreferenceに
formation 74を再生成する。target horizontalのformation列は読まない。保存済み40 selectorとexp287
version 5の15 TVT modelをSHA検証し、421列で予測する。booster trainingは0。

`submission.csv`はnotebook内のsubmit APIを使わず出力し、Kaggle output取得後に検証する。

## Inference / Public LB結果

Kaggle private CPU inference version 1（id_no `127952811`）は`COMPLETE`。notebook内部runtimeは
`448.386 sec`、Kaggle log終端は`476.945 sec`だった。14,151 rows / 3 wellsについて、保存済み
40 selectorと15 TVT modelを使い、inference-time trainingは0。

- feature surface: clean 273 + outer-matched compact 74 + formation 74 = 421
- formation reference: train 773 wells（plane利用765、dense利用766）
- target formation列read: 0、trainと同名のtest 3 wellsはself-exclude
- submit-check: `PASS`（FAIL 0 / WARN 0）
- submission SHA256: `deb46704998c2365cbdb91c20acd7ffdfefe0614cb5f2deb633eb8efd0ff8ca6`
- prediction decompressed SHA256: `eea88958df27dafe595a8f14bea4df980204143b6d1f7c01e65b98069c0daebc`
- formation Parquet / logical SHA256:
  `d5363041a9a8d48fcca29e6529f3a636e3e2cd0ba2a7d98bbcccc3d53365ab80` /
  `cc974f8cc4bd3976b42767fc690a8085389d39d249d73ff3f8e6bdf0c44c9d8c`

ユーザーのscoring完了連絡後にKaggle提出履歴を確認し、最新`ref=54842141`が
`SubmissionStatus.COMPLETE`、Public LB `7.530`だった。直前ML anchor exp264の`7.562`を
`-0.032`改善し、新しいML route Public-LB anchorとする。別routeのexp082 ensemble 7.601も-0.071で
上回るが、ensemble anchorはexp082に維持する。ただしtrain-side worst-well / 悪化well数
guard FAILは維持し、LB anchor更新をtrain-side promotionとは扱わない。

## 次のアクション

即時の救済trainは行わない。exp287をML LB anchorとして保持しつつ、train-side adoptionが必要に
なった場合だけ、保存済みexp287成果物を使う0-boosterのformation tail attribution readoutを低優先で
再検討する。実装・実行は別途ユーザー確認後とする。

routeは`ml_model`へ修正した。親exp264のPF/HMM/Beam候補は補助compact meta featureとしてのみ利用し、
direct blendやhard-pathを行わず、formation add-only後の最終予測はdownstream LightGBMが生成するためである。
