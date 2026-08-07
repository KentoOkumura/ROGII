# タスクリスト

## TODO

- 同じ親 / familyの直近実験を分類し、representation auditの発動条件を確認する。
- steeringからexp515を作成する。
- Jupytext percent形式でself-contained train/inferenceを実装する。
- joint-state exact forward-backwardを小型synthetic brute forceと照合する。
- fold-safe atlas、3-family weights、projection、hidden-test ID整列のcontract testを作る。
- static/Jupytext/strict validationとlocal debug smokeを通す。
- CPU/GPU Active Sessionsを確認してKaggle CPU trainをpushし、OOFを記録する。
- 同じfixed configでinferenceをpushし、output取得とsubmit-checkを行う。
- `LATE SUBMIT`を明示したmessageで1回submitし、monitorしてLBを記録する。
- Kaggle push 前に metadata と bootstrap 内 config の整合を確認する。
- output 取得後に feature content SHA、prediction SHA、submission SHA、model SHA を記録する。
- 実装完了時に手法契約とコードの差分を再監査する。
- negative resultが閉じるtupleと、残ったpositive submetric / oracle headroom / coverage / 誤差非相関性を `result.md` に記録する。

## 進行中

- Kaggle UI Active Sessions確認待ち。空き確認後にCPU trainをpushする。

## ブロック中

- なし

## 完了

- 依頼原文と一次資料から手法契約を抽出した。
- `input / target / output / loss / decode / context unit`を記録した。
- 元コードと一致しない近似実装であるため、実装区分を`proxy`へ訂正した。
- 実験名、提出前に固定する条件、再現性設計を確定した。
- exp515 train/inference source、正規Notebook、contract testsを実装した。
- 専用10 tests、共通込み15 tests、py_compile、Ruff、Jupytext、strict validationをPASSした。
- canonical private CPU train/inference packageを生成し、slug/title/config metadataを確認した。
