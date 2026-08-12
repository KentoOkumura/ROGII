# タスクリスト

## TODO

- 正規Notebook採用について別承認を得る。
- code submissionは全gate PASS後に別承認を得る。

## 進行中

- なし。

## 完了

- source version 2 / run 337064157をpullし、Notebook/code-cell/input/model SHAを監査した。
- ユーザー変更に従い最終式を`0.50 * exp413 + 0.50 * hjyact_v2_final`へ固定した。
- active source pathを抽出するgeneratorと別名compact self-contained inference候補を実装した。
- exp413のhidden-safe保存model runtimeを埋め込み、static prediction sidecarを禁止した。
- deterministic shared DAG、route-specific PF adapter、generation/hit trackerを実装した。
- component CSV boundary、dynamic ID/finite/duplicate、visible post-hoc SHA、formula、external-submit gateを実装した。
- `ensemble_contract.yaml`、`model_manifest.yaml`、専用契約テストを作成した。
- variant 1、LightGBM config 0、Ridge 5 fits、新規booster 0、保存model 88ファイル/108推定器を記録した。
- Jupytext round-trip、`py_compile`、Ruff F821、専用pytest 6件、`validate-exp`をPASSした。
- candidateのKaggle package/push/run承認を2026-08-05に得た。
- version 1--5をKaggle GPU/internet offで実行し、version 3--5でhjyact source parityをPASSした。
- version 5でK16をHaswell subprocessへ隔離し、runtime-fit/pinned係数監査をPASSした。
- 診断CSVのrow/header/ID順/finite/duplicateをsample submissionへ照合し、形式チェックをPASSした。
- ユーザーがmax absolute `0.02 ft` / RMSE `0.001 ft`の数値許容を選択した。
- version 6をKaggle GPU/internet offで完走し、hjyact exact SHA、exp413 numerical witness、
  reuse manifest、fixed formula、current-test submit-checkをすべてPASSした。
- version 6 submission SHAがversion 3診断runとbyte-identicalであることを確認した。
- 2026-08-05にユーザーから、上記runtime短縮3点の適用と全well inference runの明示承認を得た。
- SP45およびexp413 HMM/PF/K16のwell単位4並列化、model-package correction無効化を実装し、静的gateをPASSした。
- version 7を全well対象で完走し、version 6比23.771%短縮、component/final parity、submit-checkをPASSした。
- pre-v7 pullに保存されていたexact v6 sourceをcanonical kernel version 8としてpushし、latest versionへ戻した。
- version 8を完走し、hjyact/exp413 componentとfinal submissionがversion 6とbyte-identical、reuse/formula/
  submit-checkがPASSであることを確認した。same-v6 visible output reproducibility gateは2/2 PASS。
- v6/v8でdeterministic-GR intermediate content SHAが不一致であることを監査し、full intermediate byte
  reproducibilityとhidden-well stochastic determinismを未証明として残した。competition submitは行っていない。

## ブロック中

- なし。formal hidden deterministic anchorにはhidden-well RNG監査が残る。
