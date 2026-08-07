# exp259_coordinate_equivariance_path_warp_augmentation

## 状態

Kaggle CPU transform audit version 1と、`md_stretch`を除外したfull-well exact TVT datum
学習version 1は完了しました。学習とequivariance guardは正常ですが、exp251 clean controlとの
比較でhidden-like 2面と最大well回帰guardがFAILしたため、train-side rejectedです。
inferenceとsubmissionは実行しません。

- transform audit: `kentookumura/exp259-coordinate-equivariance-warp-audit-train`
  version 1 / ID `127328846`
- exact datum train: `kentookumura/exp259-exact-datum-fullwell-train`
  version 1 / ID `127436131`
- route: `ensemble`
- active variant / config / fold / booster: `1 / 2 / 5 / 10`
- selected schema: 295 features
- clean OOF: 3,783,989 rows / 773 wells
- synthetic: stable 193 wells（24.97%）、outer-trainのみ

## 仮説

absolute TVT datumを一貫してshiftしたtraining viewをclean rowsへ少量追加すれば、候補rankerが
datum表現へ過適合するのを抑え、clean OOF、特にlong-tailを改善できると仮定しました。

## 親実験と変更点

親はexp251のraw-test-safe 295列dual-objective rankerです。candidate bank、features、fold、
sampling seed、LightGBM設定を固定し、outer-trainへのexact TVT datum view追加だけを変更しました。
clean controlはexp251の保存済みOOFを使い、exp259では再学習していません。

## 結果

fixed-Viterbi RMSEはexp251 clean controlの`8.502212005`から`8.427125551`へ
`-0.075086454`改善しました。candidate loglossは`-0.000026563`、distance 1000+ RMSEは
`-0.082177424`で、これら3 guardはPASSです。

一方、exp115 spatialは`+0.067319432`、typewell-purgedは`+0.045377256`悪化し、
`aed44918`の最大well回帰は`+6.370552990`でした。最大許容値は`+0.25`なので、
全guard通過条件を満たしません。

## 実装契約

- clean rowsを保持し、SHA256で固定した25% wellsへ`-40/-20/+20/+40 ft`のexact datum
  viewをouter-trainだけに追加。
- validationは常にclean view。
- exp251 version 3のselected 295列schema SHA
  `7a9217d6ed96f5f1e569dbefff2a1fb17751405d6ddccae5e5d9dbf12da787ae`を固定。
- exp251 clean controlは別kernelの保存済み結果を参照し、exp259内で再学習しない。
- `md_stretch`、TVT shear、XY tilt、spline warp、smooth XYZ perturbationは学習無効。
- heel translation、reflection、yawは295列でfeature-identicalになるためaudit専用。
- 5/5 foldsでcandidate error／within10 label／相対特徴288列の不変性を確認。

## 検証方針

全773 wellsのclean GroupKFold OOFを評価し、exp251の同一295列saved controlとfixed-Viterbi
overall、candidate logloss、1000+、exp115 hidden-like 2面、最大well回帰を比較しました。
全guard通過だけを採用条件とし、outer-validへsynthetic viewを混ぜていません。

## transform audit

773 wells、9 transforms、6,957 viewsを監査しました。厳密4変換は全件PASSし、inverse最大誤差は
`9.313225746154785e-10`です。近似4変換は94.83〜99.61%採択しましたが、`md_stretch`は
trajectoryとMDの不整合により773/773 viewsが分布guardでrejectされました。

## 主要ファイル

- `config.yaml`: transform、10-booster学習、success guard、再現性契約。
- `src/coordinate_path_augmentation.py`: transform／inverse／GR resample／reject engine。
- `*_compact_selfcontained_train.py/.ipynb`: transform audit notebook。
- `*_train_variant0.py/.ipynb`: full-well exact datum学習notebook。
- `*_compact_selfcontained_inference.py/.ipynb`: inference停止guard。
- `metrics.json`: audit、学習結果、exp251との事後比較。
- `SESSION_NOTES.md`: 実装、実行、SHA、最終判定。
- `result.md`: 指標の解釈と採否。

## 所見

exact datum augmentationはoverallと1000+に改善信号を示しましたが、hidden-likeとwell-level
安定性を損ないました。採用guardを緩めず不採用とし、推論・提出へ進みません。再訪する場合も
今回のvariantを重複学習せず、long-tail限定・低synthetic比率の別variantとして原因分離します。

## 次のアクション

exp259のinferenceとsubmissionは行いません。再訪前に改善well／悪化wellのtarget-free属性差を
readoutし、1000+だけへ限定できる根拠が得られた場合に限り、別variantとしてユーザー確認後に
学習します。
