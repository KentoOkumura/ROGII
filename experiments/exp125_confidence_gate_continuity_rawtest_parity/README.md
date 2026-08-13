# exp125_confidence_gate_continuity_rawtest_parity

## 状態

- ルート: pf_beam
- 状態: completed_train_side_audit_no_submit
- CV: fair shared best RMSE 11.540333945
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-25
- 親実験: exp102_confidence_gated_likpf_fallback_on_exp101

## 仮説

`exp102` と `exp112` の confidence gate は train-side で小改善したが、そのまま推論へ移植するには continuity、worst-well、raw-test 再生成の根拠が足りない。保存済み OOF を同じ shared surface で比較すれば、直接 gate として進めるか、ML feature / 診断に戻すべきかを判断できる。

## 変更点

- 新規モデルは学習しない。
- exp102 / exp112 の保存済み OOF prediction を読み、shared surface で metrics を作る。
- by-well regression、bucket、continuity、common-worst、raw-test parity checklist を生成する。
- dense/high-drift gate prediction は optional input として扱う。

## 検証方針

- Fold: 上流 OOF prediction に従う。
- Group: well
- Stratification: なし
- Leakage Check: `true_tvt` は評価にのみ使用し、gate 条件の再探索や feature 生成には使わない。

## 実行入口

- 学習 notebook: `exp125_confidence_gate_continuity_rawtest_parity_train.ipynb`
- 推論 notebook: `exp125_confidence_gate_continuity_rawtest_parity_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp125_confidence_gate_continuity_rawtest_parity`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | fair shared best RMSE 11.540333945 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 実装は saved OOF の posthoc audit に限定し、hidden test 用の予測候補を誤って作らない。
- exp112 subset と exp102 full surface を混ぜないよう、shared surface を明示する。
- exp102 best gate は shared surface で `likpf_mean` から RMSE -0.064076 改善した。
- exp112 expected-error gate は RMSE -0.031144、within10 +0.000752 と低 switch の小改善を維持した。

### 悪かった点

- exp102 best gate は within10 が -0.001638 悪化した。
- required raw-test parity checks は 2 件 missing。Kaggle が exp101 source を追加できず、manifest / schema が見つからなかった。
- continuity fail は 3 variants、well regression fail は 14 rows、最大 well regression は +12.461017。
- exp124 dense/high-drift gate artifact がなく、dense gate との直接比較は未完了。

### リスク / 注意

- shared surface は 155 wells subset のため、global CV 根拠としては弱い。
- OOF 閾値探索の過適合リスクがある。
- この実験単体では deterministic submission anchor ではない。

## 次

1. direct inference port / submit はしない。
2. exp112 の confidence signal は ML add-only feature / confidence diagnostic に下げる。
3. row-wise gate ではなく、必要なら segment selector / Viterbi 側で continuity を扱う。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
