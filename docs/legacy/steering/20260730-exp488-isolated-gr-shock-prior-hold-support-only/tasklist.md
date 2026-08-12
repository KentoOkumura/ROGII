# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Stage 1、inference、submission。Stage 0 FAILにより実施しない。
- threshold、window、output、trigger条件のsame-data救済。

## 完了

- exp488 steeringをexp482より先に分岐作成した。
- zero-shock control 0、support top32、HMM replay 32の契約を固定した。
- exp482の科学条件とno-rescue制約を維持した。
- 再現性設計を`design.md`へ記録した。
- compact self-contained train、fail-closed inference guard、専用testを実装した。
- canonical train/inference NotebookをJupytext sourceから生成した。
- 専用pytest`14 passed`、Jupytext、構文、Ruff F821/E9、
  strict experiment validationを完了した。
- strict packageを作成し、metadataと埋め込みconfigを検証した。
- 短縮canonical slugでprivate CPU kernel v1へpushし、
  Kaggle `id_no=129170127`を確認した。
- v1は32-well計算後の`numpy.bool_` gate JSON保存だけでERRORになった。
- 科学条件を変えず、generic JSON conversionと再現testだけを修正した。
- v2修正後に専用pytest`15 passed`、exp408/440/482/488回帰
  `52 passed`、構文、Ruff、format、strict validationを確認した。
- 同じcanonical kernelへversion 2をpushした。
- Kaggle version 2のCOMPLETEを確認した。
- support32の183,093行で最終trigger 0件、parent/candidate差0を確認した。
- technical / scientific gate FAILとterminal closeを記録した。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`へ結果とSHAを反映した。
