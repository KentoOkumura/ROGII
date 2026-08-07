# exp305_tempered_raw_smoothed_exact_hmm_emission

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU train v3完了、promotion gate FAILで閉鎖
- CV / Public LB / Private LB: train-side paired RMSEのみ / なし / なし
- 作成日: 2026-07-21
- 親実験: `exp304_gr_denoiser_emission_separability_readout`
- 方法論/control参照: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp304でshift emissionのMRR/top3を改善したstationary db4 level-3 SWTを、raw exact-HMM emissionへbeta 0.15だけ弱く混ぜれば、posterior-mean decoderと固定likPF blendのRMSEを改善できる。

## 変更点

- 変更は`ell_beta = 0.85 * ell_raw + 0.15 * ell_swt`の1点だけ。
- exp209 HMMのstate grid、transition、prior、known-prefix raw sigma、posterior meanを固定した。
- 保存likPFは`last_known_tvt + likpf_mean_d`としてmaterializeした。
- 1 scientific variant、773 HMM well-runs、model/LightGBM/PF/Beam/booster 0、control再実行0。

## 結果

- direct: `11.938287 → 13.218199`、`-1.279912 ft`改善、1/5 foldsでFAIL。
- fixed likPF 50/50: `10.269693 → 10.767674`、`-0.497982 ft`改善、1/5 foldsでFAIL。
- 1000+、hidden-like 2面、by-well p95、worst-wellをすべてFAIL。
- 3,783,989 rows / 773 wells、finite 100%、ID mismatch 0、silent fallback 0、runtime 15,983.840秒。
- strict baseline parityはsaved likPF側の約`3e-6 ft`差でFAILしたが、候補悪化は`0.498--1.280 ft`でありnegative decisionは変わらない。

## 検証方針

- 保存済み5 foldsと`well_id`を使うtrain-side paired RMSE。
- predictionとcontent SHAをtruth join前に凍結する。
- overall、4/5 folds、1000+、hidden-like 2面、by-well p95、worst-well、coverage、SHAの全gateをdirectとblendの両方へ適用する。

## 再現性

- Kernel: `kentookumura/exp305-tempered-raw-swt-exact-hmm-emission-train` v3、`id_no=128079137`
- prediction content SHA256: `86b1768f18d31ba296774054c14b24e2e4650ddc74d9858d9baea8b534027302`
- scientific contract SHA256: `343084494621e1a3bb15899f6c3c441507f4dfd3af671e8de7c08f3f8867bd1b`
- 詳細なfold/scope/SHAは`result.md`と`metrics.json`を正とする。

## 結論

SWT tempered emissionはwrong modeを安定化し、directとblendを広く悪化させた。事前登録どおりbeta/sigma/HMM/blend救済、案4、raw-test inference、submissionを行わず閉じる。

## 所見

exp304のshift separability改善はexact-HMM decoder改善へ転移しなかった。単一foldの小改善に対してpooled、long-tail、hidden-like、tail-riskの悪化が一貫して大きいため、同一emission familyの追加探索は行わない。

## 次

exp305はnegative resultとして閉じる。案3/案4、beta/sigma/HMM/blend救済、raw-test inference、submissionは行わない。
