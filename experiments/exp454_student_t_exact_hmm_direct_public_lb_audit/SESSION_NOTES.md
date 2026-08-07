# exp454_student_t_exact_hmm_direct_public_lb_audit セッションノート

## 目的

tail guardで不採用となった固定Student-t exact HMMを単体Public LBで記述評価する。

## 現在の状態

- Route: `pf_beam`
- 状態: 設計確定、実装未着手
- train-side OOF: `11.720478702`
- LB: 未評価
- 実装 / Kaggle run / submission: `0 / 0 / 0`
- evaluation Notebook契約: 1物理モデルにつき1本

## コマンドログ

- 2026-07-30:
  `make new-steering EXP=exp454_student_t_exact_hmm_direct_public_lb_audit`
- 2026-07-30:
  `make new-exp EXP=exp454_student_t_exact_hmm_direct_public_lb_audit`
- 2026-07-30:
  exp374/209/389/434、`docs/06_reproducibility.md`を読み、候補、HMM状態空間、
  hidden cardinality、LB解釈、禁止事項を設計として固定した。

### 未承認の将来作業

- Jupytext起点のcompact self-contained inference実装
- Kaggle package作成・CPU実行・output取得
- submit-check
- competition submissionと監視

## 変更点

- 新しいpredictionはまだ作っていない。
- exp374候補を直接LBへ露出する別実験として採番した。
- pinned parent source/config SHAを記録した。
- 凍結提出順を3候補中3番目にした。

## 実行量契約

- scientific variants: 1
- HMM well-runs: dynamic test well数と同数
- Gaussian / Huber / PF / Beam runs: `0 / 0 / 0 / 0`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent-control再実行: 0

## 再現性メモ

- seed policy: no RNG、fixed sorted well/row/grid/rate order
- stochastic components: なし
- CPU/GPU runtime: CPU-only / GPUなし / internet off / 上限30,600秒
- Kaggle kernel id:
  `kentookumura/exp454-student-t-hmm-direct-lb-audit-inference`
- input / schema / content SHA: 実装・実行後に記録
- model manifest / model SHA: 非該当、model count 0を記録
- prediction SHA / submission SHA / rerun: 未実行

## 次のアクション

1. ユーザーの実装承認を待つ。
2. 承認後も1候補・1inference Notebookの設計を変更せず実装する。
