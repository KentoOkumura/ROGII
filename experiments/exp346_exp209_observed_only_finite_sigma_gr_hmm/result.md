# exp346_exp209_observed_only_finite_sigma_gr_hmm 結果

## 状態

Kaggle CPU canonical kernel version 1でtrain-side監査を完了した。technical gateはPASSしたが、事前定義したscientific AND gateをFAILした。判定は`observed_only_finite_sigma_failed_close_without_rescue`、実験状態は`train_side_observed_only_finite_sigma_gate_failed_closed`である。raw-test inference、submission、同一prediction上の救済は行わない。

## 仮説

raw missing行をexp209幅のまま保ち、raw finite行だけfinite-prefix population stdへ狭めれば、exp307の全行過信を避けながらGR識別力を利用できると仮定した。

## 変更点と実行量

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 単一変更: raw finite evaluation行だけGR emission scaleをwell別finite-prefix population stdへ置換
- raw missing evaluation行: exp209 zero-fill std、補間GR、Gaussian emissionを完全維持
- 1 variant / 773 HMM well-runs / 3,783,989 prediction rows
- model / LightGBM config / trained fold / booster / PF / Beam / control再実行: `0 / 0 / 0 / 0 / 0 / 0 / 0`
- Kaggle kernel: `kentookumura/exp346-observed-only-finite-sigma-gr-hmm-train` version 1、id_no `128227279`、CPU、internet off
- runtime: 17,757.849秒（4.9327時間）

## Technical gate

| 項目 | 実績 | 条件 | 判定 |
| --- | ---: | ---: | --- |
| prediction rows / wells | 3,783,989 / 773 | 3,783,989 / 773 | PASS |
| HMM runs | 773 | 773 | PASS |
| finite coverage | 1.0 | 1.0 | PASS |
| ID mismatch | 0 | 0 | PASS |
| scale fallback | 0 / 773 wells | `<=10%` | PASS |
| raw missing emission差 vs exp209 | 0.0 | 0.0 | PASS |
| posterior正規化最大絶対誤差 | `2.8866e-15` | `<=1e-6` | PASS |
| exp209 exact-HMM parity差 | `1.2566e-11 ft` | `<=1e-5 ft` | PASS |
| exp209 LikPF parity差 | `3.2766e-6 ft` | `<=1e-5 ft` | PASS |
| exp209 fixed 50:50 parity差 | `3.6416e-6 ft` | `<=1e-5 ft` | PASS |
| runtime | 17,757.849秒 | `<=30,600秒` | PASS |

raw mask partition、scale clip、truth late-join、prediction/schedule SHA freezeもPASSした。したがってnegative resultは入力・実装・baseline再生の失敗ではない。

## Scientific gate

### Direct exact-HMM

| scope | exp209 control RMSE | 候補RMSE | 改善量（ft） |
| --- | ---: | ---: | ---: |
| overall | 11.938287 | 13.295027 | -1.356739 |
| fold 0 | 10.923776 | 10.590666 | +0.333110 |
| fold 1 | 12.302481 | 12.863221 | -0.560740 |
| fold 2 | 11.570050 | 13.373852 | -1.803802 |
| fold 3 | 12.723861 | 14.094581 | -1.370720 |
| fold 4 | 12.067702 | 15.051705 | -2.984002 |
| raw GR observed | 11.933740 | 13.580807 | -1.647067 |
| raw GR missing | 11.948064 | 12.658429 | -0.710366 |
| missing fraction high | 11.792411 | 12.242958 | -0.450547 |
| MD since 1000+ | 13.135431 | 14.552170 | -1.416738 |
| hidden-like spatial | 12.564491 | 14.294024 | -1.729533 |
| hidden-like typewell-purged | 12.367244 | 13.870159 | -1.502915 |

overallは必要な`+0.05 ft`改善に対して`-1.356739 ft`、改善foldは必要な4/5に対して1/5だった。必須non-regression scopeはraw missing、high missing fraction、1000+、hidden-like 2面のすべてがFAILした。

### Fixed LikPF 50:50 guard

| 項目 | exp209 control | 候補 | 差 |
| --- | ---: | ---: | ---: |
| overall RMSE | 10.269693 | 10.531118 | +0.261425 ft |
| 改善fold | - | 2/5 | - |

fixed blendでも悪化したため、exp346 candidateを既存LikPFへ混ぜる根拠はない。

### Well別tail

- 改善380 wells、悪化393 wells。
- candidate / controlのwell別RMSE p95は`26.588420 / 25.425747`、差は`+1.162673 ft`。
- worst well `be83e781`は`3.358579 → 74.124997 ft`、回帰`+70.766418 ft`。固定上限`+0.25 ft`を大幅超過した。
- best well `91db7070`は`47.654411 → 2.436826 ft`、改善`45.217585 ft`で、well間の効果が極端に不安定だった。

scientific AND gateはFAILである。

## Scale readoutと失敗原因

- exp209 zero-fill sigma中央値: `38.641808`
- observed finite std中央値: `13.895676`
- `sigma_observed / sigma_exp209`中央値: `0.369701`
- sigma縮小幅中央値: `24.852616`
- observed finite std下限10へのclip: 92 / 773 wells
- fallback: 0 / 773 wells

raw observed行のGR emissionを中央値でexp209幅の約37%まで狭めたことにより、GR evidenceが過信された。これはexp307 finite-std全行適用の悪化`-2.271430 ft`よりは小さいが、exp346でも`-1.356739 ft`と大幅な回帰である。

raw missing行のrow-wise emission schedule自体はexp209と厳密一致したにもかかわらず、raw missing scopeも`-0.710366 ft`悪化した。exact-HMMは同一wellの全時点を平滑化するため、observed行で変えた強いevidenceがstate posteriorを通じてmissing行へ波及したと解釈する。missing行のscaleだけ固定しても、全系列posteriorの安全性は分離できなかった。

また、well別sigma比とRMSE deltaのSpearman相関は`0.1441`、sigma縮小幅との相関は`-0.1895`に留まり、単純なscale ratio thresholdでtailを安全に分離できる根拠も弱い。結果を見てthresholdを選ぶ救済は行わない。

## 再現性証拠

- promotion gate raw SHA256: `eaad34eb1889e67d9b623a67d15094349fa756180b91ba2c503e78d6bb0f4c96`
- overall/fold/scope metrics raw SHA256: `ada0d67a8c5b7facbd171b3b14a090dd83247da45a0dcb5a73cf99db911671db`
- by-well metrics raw SHA256: `ea5764ea416d6e4b838034b0267914b36b813a682f93e7987094bd68148bfb88`
- input control manifest raw SHA256: `9f5be9dd048a5a3f9bd5b1f242e48bc8da8d1f5d5f8c3d9d14f2bde7d4fd4e10`
- scientific contract declared / raw SHA256: `a0ab7df28b796cb766dbd01446e4d705f2188a76947294700e036eef5eab4e93` / `146972b7c9cf50e214d5f4aeffea4058ee9cee7a60920618946567d685e65472`
- summary raw SHA256: `15d9e68c3b6168f5dbeb8851be3dbdac9bb9fc6f580065f29146c97f9af5039b`
- prediction raw / decompressed SHA256: `9a3f97ebf11edf8a2df0415d4a24ac18bec539704d2c49961b29ce24f1251172` / `f6888ff8755d64fa72a3d8b23f0949a72b9b448b98eea898f9d9b028326fc74b`
- raw mask scale schedule raw / decompressed SHA256: `5493da7db757813366713bba31ab9bfbabad91a66f51e46ece643cfb86cccbc8` / `79473f9e56ebc5aa6b61b54f219603c2f175c9d1bfc296ad081844b40ec62796`
- scale audit raw / decompressed SHA256: `d01f02f8fbb94d4145ef8a4adfa999571e921f5b9f7b4ce46e2777d13500b584` / `892b49f20941fec7c91a79f6e46fb0b5b001a74c17731c10e5e404288485b067`
- model / submission SHA: 非該当

small metrics/gate/manifest出力だけを一時領域へ選択取得し、raw SHAとgzip展開後SHAをKaggle summaryに照合した。86.9 MBのpredictionと21.0 MBのscheduleはダウンロードせず、Kaggle summaryに記録されたfreeze SHAを根拠とする。

## 結論と次のアクション

事前契約どおりbranchを閉じる。sigma multiplier/clip、confidence/threshold、emission、HMM、blendによる救済、raw-test inference、submissionは行わない。実装済みexp346は`KAGGLE_DIRECTION.md`のバックログから削除する。

同familyの新しい救済backlogは追加しない。次は既存の独立0-booster候補`exp340_exp226_depth_alias_block_confidence_readout_on_exp264`をP1--P2として維持する。GR evidenceの重複過信を再訪する場合も、直接sigmaを狭めず、既存の低優先Stage 0 `exp343_acf_effective_sample_likelihood_tempering_audit`でknown-prefix ACFの安定性を先に検証し、gate通過と別承認なしにHMMへ進めない。
