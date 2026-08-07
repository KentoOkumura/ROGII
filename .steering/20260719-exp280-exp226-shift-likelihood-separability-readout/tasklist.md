# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし

## 完了

- `docs/06_reproducibility.md`を読み、再現性設計を`design.md`へ記入した。
- exp248の固定shift値、exp252のblock契約、exp209/279のGaussian emission、exp279の
  persistent-offset定義を設計根拠に固定した。
- 実行契約をaudit variant 1 / LightGBM 0 / trained fold 0 / booster 0 / HMM 0に固定した。
- self-contained train sourceとfail-closed inference sourceを実装した。
- target-free score freeze、truth-only readout、stable shuffled、fold/scope/shift/by-well、
  persistent episode、manifest、summary保存を実装した。
- unit test 6本、py_compile、Ruff、Jupytext変換/test、strict experiment validationを通した。
- `experiment_summary.md`へexp280を追加し、`KAGGLE_DIRECTION.md`の未着手backlogを
  「実装済み・Kaggle train待ち」へ移動した。
- project/experiment validation、canonical Kaggle package prepare、loose/bootstrap SHA一致を確認した。
- repository全体testは166 PASS。既存exp264 status進行に起因する無関係な1 FAILを記録した。
- ユーザー承認後にcanonical private CPU kernel version 1をpushし、456.972秒で完了した。
- input/score/readout/manifest SHAをoutputで照合し、row/score coverageとfold/scope readoutを記録した。
- top1/top3/MRR/signがstable shuffledを全5 foldsで上回り、固定separability guardがPASSした。
- inference / submissionを実行せず、後続residual-offset HMMを検討可能な状態へ更新した。
