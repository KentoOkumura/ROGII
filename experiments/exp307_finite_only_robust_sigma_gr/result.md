# exp307_finite_only_robust_sigma_gr 結果

## 状態

Kaggle CPU train version 2は正常完了した。finite std diagnosticとfinite MAD primaryはdirect exact-HMM、fixed LikPF 50:50 blendともに悪化したため、事前登録どおり救済せずnegative resultとして閉じる。inferenceとsubmissionは行わない。

## 設定

- 親/control: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- diagnostic: finite known-prefix GR residualのpopulation std
- primary: `1.4826 * MAD`
- fallback / clip: 20 pair未満30、`[10, 60]`
- 変更しない部分: evaluation GR補間、typewell、HMM grid/transition/prior/posterior mean
- 実行量: 2 variants、1,546 HMM well-runs、control再実行0、model/LightGBM/PF/Beam/booster 0

## 結果

| variant / 比較 | candidate RMSE | control RMSE | 改善量 | 改善fold | 判定 |
|---|---:|---:|---:|---:|---|
| finite std / direct | 14.209718 | 11.938287 | -2.271430 ft | 0/5 | FAIL |
| finite std / fixed LikPF 50:50 | 10.767490 | 10.269693 | -0.497797 ft | 1/5 | FAIL |
| finite MAD / direct | 15.661341 | 11.938287 | -3.723054 ft | 0/5 | FAIL |
| finite MAD / fixed LikPF 50:50 | 11.187333 | 10.269693 | -0.917640 ft | 1/5 | FAIL |

finite MAD directのfold別candidate-control RMSE差は`+3.746712 / +1.161175 / +3.790092 / +5.526046 / +4.123333 ft`、fixed blendは`+0.970488 / -0.381148 / +0.846989 / +1.505009 / +1.537513 ft`だった。directは全fold悪化し、blendもfold 1以外で悪化した。

primary MADの必須stress scopeもすべて悪化した。

| scope | candidate-control差 |
|---|---:|
| MD since 1000+ | +3.968383 ft |
| hidden-like spatial | +2.559348 ft |
| hidden-like typewell-purged | +2.348994 ft |
| by-well RMSE p95 | +5.235247 ft |
| worst well (`3932faa6`) | +77.405013 ft |

well単位ではfinite stdが327改善 / 446悪化、finite MADが299改善 / 474悪化だった。

## scale監査と解釈

known-prefix GR欠損率は平均23.60%、中央値20.31%だった。欠損を0で埋める旧scale中央値38.6418に対し、finite stdは13.8957、finite MADは10.1367まで低下した。finite stdは772/773 wells、finite MADは773/773 wellsで旧scaleより小さく、MADは365/773 wellsで下限10に張り付いた。fallbackは両variantとも0だった。

したがって0補完はscaleを強く膨らませていたが、その膨らみを単純に除くとGR emissionが過度に鋭くなり、誤ったalignment modeへの過信が増えたと解釈する。robust MADはstdよりさらにscaleを縮め、悪化も大きかった。同一OOFでsigma、clip、affine、likelihood、HMM、blend weightを救済調整する根拠はない。

## 技術監査

- status: `KernelWorkerStatus.COMPLETE`
- runtime: `27,402.239090 sec`（約7時間36分42秒）、上限30,600秒以内
- rows / wells / HMM runs: `3,783,989 / 773 / 1,546`
- finite coverage / ID mismatch: `1.0 / 0`
- posterior normalization max error: `2.89e-15`
- truth attachment: prediction/scale content SHA凍結後

strict technical gateはraw exact-HMM parityを`1.26e-11 ft`差でPASSした一方、正しい`last_known_tvt + likpf_mean_d`復元後のsaved LikPFと50:50 controlが事前値から`3.28e-6 / 3.64e-6 ft`ずれ、固定許容値`1e-6 ft`を超えたためFAILした。この差は科学的悪化`0.498--3.723 ft`より十分小さく、全fold/scopeでのnegative decisionを変えない。positive resultの根拠にはできないが、候補棄却は信頼できる。

## 再現性

- Kaggle kernel: `kentookumura/exp307-finite-only-robust-sigma-gr-train` version 2、`id_no=128085112`
- scientific contract SHA256: `abb340cee25878ede3c87a0017e02920952d1ba01680748b36010430387f6ce2`
- input/control manifest SHA256: `0382500b48ec41ee30a91bc6b843e7cb602dcf590ba561eb58a9296f6c67f8fc`
- prediction raw/content SHA256: `76bacfb2...3486` / `8138303b...e522`
- scale audit raw/content SHA256: `2d1b6f8b...94aa` / `edde07fb...e19`
- promotion gate SHA256: `a8f134f6...68e`
- overall/fold/scope metrics SHA256: `c02cd669...bee5`
- by-well metrics SHA256: `6edb4b82...a0e`

fold/scope、gate、scale、by-wellなど小型生成物だけを選択取得し、Notebook summary記録SHAとの一致を確認した。147 MBのprediction本体は取得せず、Notebookが出力したraw/content SHAを記録した。

## 次

exp307はnegative resultとして閉じる。事前登録のparent PASSを満たさないため、exp308、exp309、exp310と、その固定lineageに依存するexp323--exp328は未実行のまま閉鎖する。将来別parentへ再設計する場合は、新しい独立根拠と事前設計、ユーザー確認を必要とする。inference、submission、同系救済backlogは追加しない。
