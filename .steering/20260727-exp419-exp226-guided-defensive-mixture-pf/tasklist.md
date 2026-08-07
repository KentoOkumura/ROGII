# タスクリスト

## 仮説と変更点

exp226 geometryをabsolute pathやoffsetとして使わず、PFのrate proposalへだけ導入する。
元transition 50%を残すfixed defensive mixtureと`p0/q`補正以外のPF要素は変更しない。

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 次のアクション

- terminal closeを維持する。独立した必要性と別承認がある場合だけ、
  保存済み生成物によるsupport失敗原因readoutを別実験として検討する。

## 完了

- exp419 steeringを作成。
- exp419 experiment scaffoldを作成。
- route、親、proposal式、importance correction、実行量、再現性、
  technical / mechanism / adoption gateを固定。
- HMM再デコード、absolute exp226 prior、blend、adaptive proposalを対象外に固定。
- `KAGGLE_DIRECTION.md`のbacklogへexp419をPF route P1として追加。
- ユーザーからtrain-side実装の明示承認を得た。
- exp404 exact kernelとexp226 fold-safe geometry artifactの列・dtype・SHA契約を確認した。
- Jupytext compact self-contained train候補とcompact Notebookを作成した。
- defensive mixture、log-space `p0/q`、temperature-5 aggregationを実装した。
- proposal allowlist、target-free per-seed support freeze、truth-late readoutを実装した。
- syntheticで`p0/q <=2`とgeometry-weight-0 exp404 bitwise parityを確認した。
- 4-shard strict merge、technical / mechanism / standalone adoption gateを実装した。
- 初回実装時のexp419専用test `12 passed`、Jupytext test、py_compile、F821、
  strict experiment validationを通した。
- 初回実装完了時にconfig、SESSION_NOTES、README、result、metricsへ
  実装済み・未実行の中間状態を記録した。
- ユーザーから正規train Notebook採用、package、push、train-side runの承認を得た。
- full前`preflight_probe`を実装し、対象test `24 passed`と静的検証を通した。
- 正規train Notebookへcompact候補を採用し、canonical preflight packageを作成・検証した。
- train-side full shard実行へ進む時点でもinference、submissionを承認範囲外に維持した。
- preflight version 1（id_no `128832139`）がCOMPLETEし、実ファイルtechnical gateを
  PASSした。
- shard0 / 1 version 1がCOMPLETEした。
- shard2 / 3 version 1がCOMPLETEし、4 shardの生成物が揃った。
- strict merge version 1（id_no `128974840`）がCOMPLETEした。
- technical gate PASS、mechanism / standalone adoption gate FAILを確認した。
- `proposal_rejected_close_without_same_oof_rescue`でterminal closeした。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`を最終結果へ更新した。
- inference、submissionは科学gate FAILにより非対象のまま終了した。
