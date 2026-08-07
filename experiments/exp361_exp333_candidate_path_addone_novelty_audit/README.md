# exp361_exp333_candidate_path_addone_novelty_audit

## 状態

- ルート: `ensemble`
- 状態: Kaggle CPU version 2完了・technical PASS・candidate novelty PASS
- CV: exp333 Stage 1 保存値 `9.0766766609`（参考）
- Public / Private LB: なし
- 作成日: 2026-07-23
- 親実験: `exp333_exp226_k16_segment_residual_offset_target`

## 仮説

exp333 は exp228 や exp263 を単体で置き換えるには至らなかったが、局所的に異なる誤差構造を
持つなら、exp293 の固定12候補バンクへ追加したときに H512 / whole-well oracle headroom を
増やせる。これは exp333 を候補パス改善として評価する仮説であり、単体CVの昇格判定とは分離する。

## 変更点

- exp333 Stage 1 の保存済み OOF 1本だけを add-one 候補にする。
- exp293 fixed deployable12、block assignment、float32/tie policy、exp302 novelty 閾値を固定する。
- exp226/exp228/exp263 の単体 score は文脈として記録するが hard gate にしない。
- model再学習、候補生成、blend、selector、raw-test inference、submissionは行わない。

## 固定PASS条件

- technical guard が全件 PASS。
- H512 oracle RMSE 改善 `>=0.03 ft`。
- whole-well oracle RMSE 改善 `>=0.02 ft`。
- H512 strict unique-best fraction `>=2%`。
- H512 oracle RMSE が `>=4/5 folds` で改善。

## 検証方針

- Fold: 保存済み exp333/exp226 の同一5 fold identity
- Group: `well_id`
- Score rows: train unknown suffix
- Leakage check: exp333予測・固定bank・blockをSHA freeze後にだけ真値を読む
- Direct score: 保存結果の再現確認のみ。科学的 gate には使わない

## 実行契約

- scientific candidate: 1
- reporting folds: 5
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent/control regeneration: 0
- CPU、GPUなし、inferenceなし、submissionなし

## 実行入口

- Jupytext source: `exp361_exp333_candidate_path_addone_novelty_audit_compact_selfcontained_train.py`
- 正規 train Notebook: `exp361_exp333_candidate_path_addone_novelty_audit_train.ipynb`
- inference は fail-closed のみで実行対象外。

## 所見

- exp333 の元実験は単体置換を問う設計で、候補パスとしての相補性は未評価だった。
- H512 oracleは`3.683763 -> 3.550659`（`+0.133104 ft`）、whole-wellは
  `+0.102132 ft`改善し、H512 strict unique-bestは`11.5064%`、5/5 folds改善だった。
- oracle novelty は実現可能な selector 性能ではなく、current-test 候補を作る価値の診断である。

## 次

exp333 current-test inference は exp333 内へ別承認で実装する。exp361からpredictionや
submissionは作らず、固定12への組み込み方も別途設計する。
