# 要件

## 依頼

exp238でHMM・self-GR HMM・exp226を候補へ追加したnested selectorを維持しつつ、
その35個の`nsel_*`出力をexp218の380特徴へadd-onlyしない。exp218に既にある
learned-likelihood selector出力29列を、fold-safeなnested selector出力で上書きする
replacement-only LightGBMを実装する。

## 制約

- Route: `ml_model`。候補パスはselector confidence生成にだけ使い、最終予測を直接置換・blendしない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親feature surfaceはexp218、selector score/fold contractはexp238 selector train v4を固定入力にする。
- outer-valid wellの正解TVTはselector学習にも最終LightGBM学習にも入れない。
- exp238の35 `nsel_*`列は最終LightGBMへ追加しない。
- exp218の既存selector入力診断25列（multi-observation 15列、legacy candidate delta 10列）は維持する。
- exp218 historical controlを再学習しない。exp218/exp238の保存済みOOFを参照baselineにする。

## 受け入れ基準

- LightGBM schemaはexp218と同じ380列・同じ順序・同じ列名である。
- 上書き対象29列と維持対象25列を明示し、両者の和が既存`ll_*` 54列と一致する。
- 11候補のpredicted-errorからinverse-error probabilityを決定的に作り、rank、margin、entropy、spread、legacy candidate別score、weighted TVTを既存29列へ上書きする。
- outer 5 foldそれぞれでexp238のrole=`train` nested OOF scoreとrole=`valid` inner-ensemble scoreだけを使う。
- active variant 1、LightGBM config 3、fold 5、合計15 boosters、parent/control再学習0をpush前に記録する。
- CV完了前はinference/submitをfail-closedにする。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
