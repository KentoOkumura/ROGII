# exp363_sticky_gr_reliability_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: `stage_0_failed_closed`
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-23
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Kaggle Stage 0: CPU version 1完了、technical PASS / scientific FAIL
- Stage 1 / inference / submission: 不適格・未実施

## 仮説

exp209 の GR 尤度が一時的に信用できない区間を sticky な潜在状態として周辺化すれば、
rate dynamics を変更せずに誤った観測への過信だけを抑えられる。

## 変更点

- 状態を `(position, rate)` から `(position, rate, q)` へ拡張する。
- `q={normal, weak}`、weak 時の log emission 係数を `0.25` に固定する。
- q の固定遷移以外の exp209 設定は変更しない。
- rate または rate 変化を prefix / geometry から予測する処理は含めない。

## 検証方針

- Stage 0: 保存済み exp209 path 上で 512-row block の weak posterior を truth なしで凍結し、
  truth join 後に bad-block AUC、fold、hidden-like、circular control を評価する。
- Stage 1: Stage 0 全 gate と別承認を通過した場合だけ、1 variant / 773 exact-HMM runs。
- 親 control は保存予測を使い再実行しない。worst-well 回帰上限は `+0.25 ft`。
- well 単位 5 fold。未知 suffix truth は posterior freeze 後にのみ結合する。

## 実行入口

- 学習 notebook: `exp363_sticky_gr_reliability_exact_hmm_train.ipynb`
- 推論 notebook: `exp363_sticky_gr_reliability_exact_hmm_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp363_sticky_gr_reliability_exact_hmm`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。
- 現在は`execution.run_stage_0=false`、`kaggle_push_approved=false`であり、
  完了済みversion 1を再実行しない。
- inference notebookはStage 1未実装を検知して明示停止し、sample submissionをコピーしない。

## 結果

| メトリック | 値 |
| --- | --- |
| pooled bad10 AUC | 0.607552 |
| circular bad10 AUC | 0.583996 |
| real - circular AUC | +0.023556 |
| Q4 - Q1 mean block RMSE | +4.816306 ft |
| hidden-like spatial / typewell-purged AUC | 0.546058 / 0.552195 |
| row-weighted weak mass | 0.589441 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- HMM と PF で同じ reliability 仮説を別実装として比較できる設計になっている。
- target-free block posteriorとledgerをSHA凍結してからtruthを結合する実装になった。
- technical gate、pooled AUC、circular差、四分位差、5/5 fold AUCはPASSした。

### 悪かった点

- hidden-like spatial AUCは`0.546058 < 0.55`でFAILした。
- weak massは`0.589441 > 0.50`で、reliability状態が広すぎる区間をweakと扱った。

### リスク / 注意

- qは完全縮退ではないが、weak mass上限を超えたためStage 0で終了した。
- multiplier や遷移確率の同一 OOF grid search は行わない。

## 次

branchを閉じる。Stage 1、transition/multiplier/sigma救済、推論、提出は行わない。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
