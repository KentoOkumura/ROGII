# タスクリスト

## TODO

- output取得後にsubmit-checkを通し、`LATE SUBMIT`固定messageで1回だけ提出する。
- scoringを監視し、late LB、kernel version、runtime、prediction/submission/checkpoint SHAを記録する。
- negative resultが閉じるtupleと残るpositive evidenceを`result.md`へ記録する。

## 進行中

- Kaggle inference version 2実行。version 1のraw/text config SHA guardだけを修正済み。

## ブロック中

- なし。GPU残量1.07hでの実行リスクはユーザー承認済み。

## 完了

- 6位discussionと現行公開submission Notebookを取得した。
- 公開kernel outputからv96/v97/v100 PF configを取得し、`pfA × twGR`のexact configを特定した。
- 依頼原文とユーザー追加確認から単体PFの手法契約を抽出した。
- `input / target / output / loss / decode / context unit`を記録した。
- 実装区分を`faithful`へ固定した。
- late-submit one-shot契約と再現性設計を固定した。
- steeringからexp516を作成した。
- 公開Notebookからanchor、emission、`pfA × tw`、whole-smootherをself-contained Jupytext sourceへ抽出した。
- 公開v96 `pfA` configをvendor copyし、source/config/checkpoint identity guardを実装した。
- PF状態、likelihood、600 particles、32 seeds、whole-smoother、single-bank/single-representationのcontract testを作成し、6件PASSした。
- current hidden testの動的列挙、sample ID 1対1整列、finite/duplicate/missing/extra guardを実装した。
- static、Jupytext round-trip、strict experiment validationを通した。
- canonical inference Notebookへ採用し、private / T4 / internet off / `LATE SUBMIT` titleのKaggle packageを生成した。
- 実装後の再監査で、anchorはpublic cell完全一致、emissionはpath resolverとterminal mainだけ変更、PF engineはterminal smoke mainだけ除外、数値契約のproxy化なしを確認した。
