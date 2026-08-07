# 要件

## 依頼

`bimodal_posterior_pfbeam_candidate_audit` backlog を実験として実装する。

## 制約

- Route: `pf_beam`
- Kaggle train-side diagnostic として実装し、PF/Beam 再生成、ML 学習、inference port、submit は行わない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- top2 mode 抽出と posterior weight は target-free 情報だけで行い、true TVT は scoring のみに使う。
- posterior temperature は config 固定で、same-OOF truth に合わせて選ばない。
- exp072 fixed PF/Beam / likPF candidate cache は比較 readout にだけ使う。

## 受け入れ基準

- `experiments/exp171_bimodal_posterior_pfbeam_candidate_audit/` に config、train/inference notebook source、helper、README、SESSION_NOTES、result、metrics が揃う。
- train notebook で入力確認、top2 mode / posterior audit、metrics / 生成物保存がセル単位で追える。
- 出力として row context、candidate metrics、bucket metrics、well metrics、commit 比 gain、summary JSON を保存する。
- GPU 学習コストが 0 booster として `SESSION_NOTES.md` に記録されている。
- deterministic anchor として扱わないこと、gzip 生成物の decompressed content SHA を主証拠にする方針が記録されている。
- Jupytext 変換、構文チェック、ruff F821、experiment validation が通る。
