# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260624-exp118-same-typewell-other-horizontal-prefix-gr-transfer-audit/` を作成した。その後、既存 exp118 との番号衝突を確認し、`docs/legacy/steering/20260624-exp119-same-typewell-other-horizontal-prefix-gr-transfer-audit/` に改番した。
- `experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/` を exp109 から作成した。
- exp119 用の `config.yaml`、補助 `.py`、train notebook、inference notebook、README、SESSION_NOTES、result、metrics を更新した。
- 再現性設計を `design.md` に記入した。
- stochastic 処理を追加せず、random control は SHA256 由来 index で deterministic にした。
- Python 構文チェックを通した。
- ruff check を通した。
- `validate-exp` を通した。
- 小さい local smoke で `run_audit` が完走することを確認した。正式 CV ではない。
- Kaggle train package を生成し、metadata と generated helper の py_compile / ruff を通した。
- Kaggle train v1 を実行し、`DeadKernelError: Kernel died` を確認した。
- v2 で source / candidate grid を軽量化し、generated package の py_compile / ruff を通した。
- Kaggle train v2 を同じ kernel id に push した。
- Kaggle train v2 の logs と主要 metrics output を取得した。
- v2 結果は baseline `likpf_mean` が best で、same-typewell GR transfer は negative controls より弱かったため rejected no-submit とした。
