# タスクリスト

## TODO（別途判断・承認後）

- 正規Notebook採用の承認を得る。
- source RNG semanticsを維持するか、well別明示seed candidateを作るか判断する。
- seed固定を選ぶ場合は、visible final parityと未知well相当の中間2-run一致を再検証する。
- hidden RNG方針と提出準備が承認された後、`kaggle-submit-check`を実行する。
- competition submissionの承認後、code submissionを1回行い、scoringを監視する。
- Public LB、submission ref、runtime、SHA、解釈を実験記録、サマリー、提出履歴へ反映する。
- 結果後に`KAGGLE_DIRECTION.md`からexp513を削除し、exp512の実行優先度を再評価する。

## 進行中

- なし

## ブロック中

- hidden code submissionはpre-override PF/Ridge blendのrun差によりfail-close中。
- 正規Notebook採用、submit-check、competition submissionは未承認。

## 完了

- exp512のsource identity、final boundary、実行量、再現性契約を監査した。
- exp512 v1/v2の最初の意味のあるtracebackを読み、旧competition mountと旧Ridge dataset mountを原因と確定した。
- source Notebookとexp512 v3 generatorのSHAを固定するexp513専用generatorを作成した。
- 完全なhjyact-v2 final成分だけを抽出したJupytext percent sourceを作成した。
- 7章構成、48 cellsの別名compact self-contained inference Notebook候補へ変換した。
- exp413、50/50 blend、cross-consumer candidate reuseが入っていないことを契約testで固定した。
- competition rootとRidge rootを内容検査・SHA監査で一意解決し、旧path直接assignmentを禁止した。
- source boundary、input/model inventory、dynamic ID/order、finite/duplicate、post-hoc SHA、seed契約testを作成した。
- candidate sourceを5,097行 / 236,961 bytesへ縮小した。
- Jupytext round-trip、`py_compile`、Ruff F821、専用test 7件、親込み13件、`validate-exp`をPASSした。
- 実行量をvariant 1、LightGBM config 0、新規booster 0、親再学習0、Ridge 5 fits、
  saved model 13 / estimators 33として`SESSION_NOTES.md`へ記録した。
- 正規Notebookを上書きせず、実行専用`current_test_inference`を使った。
- 437,126-byteの最小bootstrap packageを作り、metadata、embedded SHA、1 MiB制限をPASSした。
- private canonical kernelへ同一packageをversion 1 / 2としてpushし、両方COMPLETEまで監視した。
- exp512 v1/v2のcompetition/Ridge mount failureが再発せず、Ridge 5 foldと保存model推論が完走した。
- 両runの14,151行visible finalがsource SHA`b192d3f3...9ded4a`と一致した。
- version 1 / 2の科学runtime`819.939 / 816.881`秒、input/model/final SHAを記録した。
- pre-override blend統計のrun差と、unseeded Numba PF + threaded well generationを特定した。
- Kaggle CLIの旧version output指定不具合を確認し、誤取得したlatest複製を隔離した。
- competition submitは行っていない。
