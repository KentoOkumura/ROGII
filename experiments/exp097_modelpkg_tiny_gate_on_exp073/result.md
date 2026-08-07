# exp097_modelpkg_tiny_gate_on_exp073 結果

## 仮説

exp073 base prediction と model-package prediction が近い row だけ、ごく小さい agreement gate で model-package 側へ寄せると、direct replacement のリスクを避けながら微小補正候補を作れる。

## 設定

- 親: exp073_gpu_reproducibility_guard_for_exp063_full_replay
- 参照: pilkwang/rogii-target-free-tvt-geosteering
- 検証: submission diff guard
- メトリック: raw modelpkg diff、correction magnitude、prediction range、SHA
- シード: 42。ただし本実験は RNG なし

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | 8.766 |
| Private LB | - |
| 実行状態 | Kaggle inference v3 completed / submit-check PASS |
| selected variant | `modelpkg_gate_g005_s4p0` |
| raw diff p95 | 33.566870 |
| correction p95 | 0.008829 |
| correction max | 0.010000 |
| submission rows | 14,151 |
| submission SHA | `9467e55c136d09063a284d8b31cf412a66130963b07642d28194e303d8ac2175` |
| submit ref | `53897072` |

## 再現性

- deterministic anchor: false
- seed policy: no_new_rng_submission_diff_postprocess
- kernel version: `kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference` v3
- feature content SHA: exp073 decompressed content `f3f04e56f3035191d651e330d26ee48e819f42cf0497acecefc88fe985cdc219`
- model SHA / manifest SHA: 対象外
- prediction SHA: `3250801c6937e3deb77d35b5aea1a3f2bcbf8cf10eca7ecfb24080f11c6f7e0e`
- submission SHA: `9467e55c136d09063a284d8b31cf412a66130963b07642d28194e303d8ac2175`
- rerun result: Kaggle inference v3 completed

## 解釈

Kaggle inference v3 が完了し、selected `gmax=0.005, scale=4.0` は公開 rows で guard を通過した。補正量は非常に小さく、最大でも約 0.01 ft なので、予測をほぼ exp073 のまま維持する設計になっている。`submission.csv` は 14,151 rows で submit-check PASS。

code submission ref `53897072` は Public LB 8.766 で完了した。exp073 raw 8.780 よりわずかに良いが、exp077 8.611 / exp096 8.651 より悪いため anchor にはしない。

ただし OOF surrogate がないため、この結果は CV 改善の証拠ではなく、exp073 inference と model-package-only prediction の submission diff guard として扱う。

v1 code submission は hidden rerun で `Notebook Threw Exception` になった。原因は公開 output copy 型の入力依存で、hidden の `sample_submission` と公開 `submission_model_package_only.csv` が行一致しないため。v3 では exp073 base を現 test rows で再生成し、model-package CSV が現 sample ids を cover しない場合は exp073 base-only submission に fallback する。

## 次

提出する場合は `kentookumura/exp097-modelpkg-tiny-gate-on-exp073-inference` v3 を code submission として使う。hidden で model-package correction を本当に効かせるには Pilkwang model package branch の直接再生成 port が必要。
