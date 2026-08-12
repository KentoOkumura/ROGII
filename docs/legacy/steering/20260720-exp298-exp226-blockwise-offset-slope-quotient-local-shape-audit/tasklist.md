# タスクリスト

## TODO（別承認が必要）

- なし。本枝はscientific guard FAILにより閉じた。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `exp298`を次の空き番号として確認した。
- steeringを実験ディレクトリより先に作成した。
- route、親、比較bank、成分、quotient数式、horizon、scope、freeze順序を固定した。
- technical/scientific PASSとFAIL時closeを固定した。
- 案2・3・4を`downstream_branch_contract.md`へ固定した。
- Lateフェーズを対象外と明記した。
- `docs/06_reproducibility.md`に沿うSHA記録方針を設計した。
- target-free exp226 allowlist loader、component reconstruction、exp293 bank/block SHA gateを実装した。
- component/freeze manifestとpost-freeze raw-train truth loaderを実装した。
- H128/H256/H512/whole-wellのoffset/affine quotient、一次差/二次差、fold/scope/block/by-well readoutを実装した。
- oracle offset/slope係数と補正predictionを生成・永続化しない集約実装と専用testを追加した。
- compact self-contained train候補とfail-closed inference候補を別名Jupytext `.py/.ipynb`で作成した。
- 専用test、構文チェック、Ruff F821/E9を実行した。
- 既存の正規train/inference Notebookは上書きしていない。
- ユーザー承認により、exp293境界を維持したままsingletonをaffine/unique-best分母から除外し、長さ2以上へ
  coverage 1.0を要求する契約へ改訂した。
- ユーザー承認後、compact self-contained train候補を正規train Notebookへ採用した。
- `1 audit / 0 LightGBM config / 5 evaluation folds / 0 trained folds / 0 boosters / 0 PF-Beam reruns`を
  再確認し、strict package validation後にKaggle private CPUへpushした。
- version 1の監査後表示`KeyError`を、監査契約を変えず同じkernel IDのversion 2で修正した。
- Kaggle version 2をCOMPLETEし、technical guard全PASS、scientific guard FAIL、branch closeを確認した。
- 小規模metrics/manifestを選択取得し、input/bank/component/block/freeze/readout SHAとsingleton countsを記録した。
- 完了済みexp298を`KAGGLE_DIRECTION.md`のtrain待ちバックログから削除し、救済backlogを追加せず枝を閉じた。
