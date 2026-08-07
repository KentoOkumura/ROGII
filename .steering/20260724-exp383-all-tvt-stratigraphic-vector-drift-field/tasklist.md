# タスクリスト

## TODO

- なし。別計算方式は新しい実験契約として扱う。

## 進行中

- なし。

## ブロック中

- inference、submissionは別のユーザー指示まで停止。
- exp384/385はexp383 PASS artifactがないため停止。

## 完了

- 2026-07-24: `exp383_all_tvt_stratigraphic_vector_drift_field`として採番した。
- 2026-07-24: steering 3文書と実験scaffoldを作成した。
- 2026-07-24: 全TVT window、6地層surface、vector field、prefix校正、
  uncertainty shrink、path solve、Stage 0/1 gateをdesign-only契約として固定した。
- 2026-07-24: `docs/06_reproducibility.md`に沿うSHA、runtime、determinism方針を固定した。
- 2026-07-24: ユーザーの実装指示を受け、compact self-contained train候補と
  fail-closed inference候補を実装した。
- 2026-07-24: outer-fold role read guard、surface point catalog、全TVT
  multiscale donor catalog、absolute/vector field、uncertainty、全prefix校正、
  exp226縮約、banded path solve、target-free freeze、late truth joinを実装した。
- 2026-07-24: exp384向け生成物/manifest契約と16-well resource preflight modeを実装した。
- 2026-07-24: 専用contract test `14 passed`、Ruff、py_compile、
  compact Jupytext生成を確認した。
- 2026-07-24: ユーザーの実行指示を受け、正規Notebook採用、Kaggle CPU
  package/push、16-well preflight、PASS後full runの承認を記録した。
- 2026-07-25: canonical version 1はfold 0 donor surface joinで`MergeError`となり、
  truth join前に停止した。
- 2026-07-25: scale込み一意donor `query_id`へ修正し、専用test `15 passed`、
  Ruff、py_compile、Jupytext round-tripをPASSした。
- 2026-07-25: fold 0実測と全fold donor window数からsurface stageを30.52時間と
  投影し、固定8.5時間gateの3.59倍としてStage 0 resource FAILを確定した。
- 2026-07-25: version 2/full run/Stage 1/inference/submissionを停止した。
