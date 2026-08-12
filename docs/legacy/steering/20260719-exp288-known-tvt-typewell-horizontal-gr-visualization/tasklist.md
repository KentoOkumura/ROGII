# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 要件、設計、再現性方針を固定した。
- `exp288` experiment scaffoldを作成した。
- self-contained train notebookとdisabled inference notebookを実装した。
- config、SESSION_NOTES、result、metrics、READMEを診断専用契約へ更新した。
- Jupytext、compile、Ruff、strict experiment validationをPASSした。
- raw train 773 horizontal / 773 typewell / missing pair 0を確認した。
- Kaggle packageをrun-on-push falseでprepareし、metadataとbootstrap SHA整合を確認した。
- `experiment_summary.md`を更新した。
- prediction-target rowのtrain true `TVT`参照GRとtarget区間着色を実装した。
- target拡張後のJupytext、compile、Ruff、strict experiment validationをPASSした。
- synthetic known/target補間・target MD span testをPASSした。
- Kaggle packageを再prepareし、更新config/sourceのbootstrap SHA整合を確認した。
- `experiment_summary.md`をprediction-target EDA拡張後の内容へ更新した。
- private CPU、0 variant / 0 model / 0 fold / 0 boosterでKaggle train v1を実行した。
- Kaggle train v1が`COMPLETE`、773 PNG保存、skip 0であることをlogsとsummaryで確認した。
- 全Kaggle outputを`kaggle/output/train_v1`へ取得した。
- manifest 773行に対してPNGの存在、byte size、SHA256を全件照合し、不一致0を確認した。
- 代表PNG `000d7d20.png`を目視し、上下2段とprediction-target着色を確認した。
- config、SESSION_NOTES、result、metrics、README、`experiment_summary.md`をtrain v1実行結果で更新した。
