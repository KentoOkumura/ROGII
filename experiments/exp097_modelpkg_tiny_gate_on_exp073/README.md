# exp097_modelpkg_tiny_gate_on_exp073

## 状態

- ルート: ml_model
- 状態: submitted_complete_public_lb_8_766
- CV: なし
- Public LB: 8.766
- Private LB: -
- Submit ID: 53897072
- 作成日: 2026-06-21
- 親実験: exp073_gpu_reproducibility_guard_for_exp063_full_replay

## 仮説

exp073 の deterministic ML inference を主軸に保ったまま、Pilkwang model package prediction と近い row だけを最大 0.5% 程度寄せれば、大外しを抑えたまま微小な補正候補を作れる。

## 変更点

- exp073 `lgb_mean` inference prediction を base として読む。
- `submission_model_package_only.csv` を id で align する。
- `g = gmax / (1 + (abs(pkg-base)/scale)^2)` の agreement gate を適用する。
- `gmax` 0.003 / 0.005 / 0.010、`scale` 4 / 5 / 8 を diff guard で監査する。
- 選択候補は `gmax=0.005, scale=4.0`。guard 通過時だけ `submission.csv` を書く。

## 検証方針

- Fold: なし
- Group: well は reporting 用のみ
- Stratification: なし
- Leakage Check: true TVT は使わない。OOF surrogate がないため CV 根拠として扱わない。

## 実行入口

- 学習 notebook: `exp097_modelpkg_tiny_gate_on_exp073_train.ipynb`
- 推論 notebook: `exp097_modelpkg_tiny_gate_on_exp073_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp097_modelpkg_tiny_gate_on_exp073`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `EXPERIMENT_ALLOW_LOCAL=1` の明示 smoke のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | 8.766 |
| Private LB | - |
| Kaggle inference | v3 complete |
| Code submission | ref 53897072 complete |
| submit-check | PASS |

## 所見

### 良かった点

- exp073 base と model-package-only prediction を直接置換せず、補正量を guard できる実装にした。
- guard failure 時は `submission.csv` を書かない。

### 悪かった点

- OOF surrogate は未実装で、CV 改善は確認できない。

## リスク / 注意

- model package の直接再生成は未実装で、reference notebook 互換の `submission_model_package_only.csv` を読む。
- v1 は公開 output copy 型だったため、hidden code-submission rerun で `sample_submission` が差し替わると例外になる。v3 では exp073 base を現 test rows で再生成し、model-package CSV が現 sample と合わない場合は exp073 base-only submission に fallback する。
- OOF surrogate がないため、提出判断は diff guard、prediction range、submit-check 後の手動レビューが必要。
- `model-package-only` は exp081 で大きく外れており、単独置換はしない。

## 次

1. 提出する場合は code submission として `kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference` version 3 を使う。
2. Public LB 8.766 は exp073 raw 8.780 よりわずかに良いが、exp077 8.611 / exp096 8.651 より悪いので anchor にはしない。
3. hidden で model-package CSV が使えない場合は exp073 base-only になるため、真の model-package tiny gate を hidden-compatible にするには Pilkwang model package branch の直接再生成 port が必要。
