# 設計

## アプローチ

exp209 absolute-TV​T exact HMMのGaussian emissionだけを固定`df=4` Student-tへ
置換する。exp342で0-HMM shift-rank FAIL後にもfull HMMが小幅改善したため、
shift-rank proxyを実行条件にせず、将来の実装・実行承認時は1 variantを直接
773 wellsで評価する。

この実験はexp342の再実行・reparentではない。exp342はexp281 residual-offset座標、
exp374はexp209 absolute-TV​T座標を科学的親とする別仮説である。

## 実験範囲

- 対象実験: `exp374_exp209_student_t_exact_hmm_emission`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- negative reference:
  `exp342_exp226_student_t_residual_offset_emission_audit`
- 変更する変数:
  - `emission="gauss"`から`emission="t"`へ変更
  - `df=4.0`
- 固定する変数:
  - state coordinate: absolute TVT
  - grid: exp209のType Well範囲、last-known TVT、`band_pad=100 ft`
  - grid step: `0.35 ft`
  - rate states: `41`
  - rate span: exp209のzero-centered
    `max(0.10, abs(prefix_initial_rate) + 0.04)`
  - `sig_r=0.002`、`sig_p=0.02`
  - `lam=1.0`、`start_sig=0.75`、`r0_sig=0.01`、`momentum=0.998`
  - sigma mode: known-prefix zero-fill population standard deviation、
    clip`[10,60]`
  - raw evaluation GR:
    linear interpolation both directions後にType Well mean fallback
  - Type Well GR:
    TVT sort、ffill/bfill、linear interpolation、endpoint hold
  - posterior output: mean

Student-tのlog likelihoodは
`-2.5 * log1p(z^2 / 4)`とする。Student-t normalization constantは全stateで
同一なのでposterior比較から省略する。追加clip、temperature、likelihood weightは
加えない。

## 入力契約

- exp209 saved exact-HMM cache:
  - prediction column: `hmm_mean_tvt`
  - expected decompressed SHA:
    `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- exp072 saved LikPF cache:
  - prediction: `last_known_tvt + likpf_mean_d`
  - expected decompressed SHA:
    `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
- reporting fold source:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
  - expected decompressed SHA:
    `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
  - decoder利用可能列: `well_id`、`row_idx`、`suffix_offset`、`fold`
  - `tvt_geop`、`tvt_pred`、`gr_delta`はdecoderへ渡さない
- hidden-like assignment expected SHA:
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`

Gaussian parentは保存cacheをmetric/control readoutにだけ使い、HMMを再実行しない。
Candidate生成APIにはunknown-suffix TVT、error、abs_error、formation、
hidden-like role、parent/candidate metricを渡さない。

## 将来の実行フロー

1. raw input、exp209/exp072 cache、fold identityのSHAとschemaを検証する。
2. raw GR finite/missing mask、known-prefix sigma、exp209固定HMM contractをfreezeする。
3. fixed`df=4` Student-t 1 variantをsorted 773 wellsで実行する。
4. candidate path、well manifest、logical content SHAをfreezeする。
5. 保存済みGaussian parent、unknown-suffix truth、fold、distance、
   hidden-like roleをlate joinする。
6. overall/fold/observed-missing/high-missing/1000+/hidden-like/by-wellと、
   fixed LikPF/HMM 50:50を評価する。
7. technical/scientific AND gateとdecisionを保存する。

## 実行量契約

- scientific variants: `1`
- HMM well-runs: `773`
- model / LightGBM configs: `0`
- trained folds / boosters: `0 / 0`
- PF / Beam runs: `0 / 0`
- Gaussian parent HMM再実行: `0`
- GPU: `false`
- inference / submission: `false / false`
- 想定runtime: Kaggle CPUで約3.2〜5時間、上限`30,600 sec`

実装承認だけではKaggle実行を許可しない。run前に上記
`1 variant / 773 HMM / 0 booster / control rerun 0`を再確認して別承認を得る。

## 生成物契約

将来実装時は少なくとも次を`artifacts/`へ保存する。

- scientific contract JSON
- input/control manifest
- Student-t OOF prediction CSV gzip
- candidate/fold/scope/by-well metrics
- raw observed/missingおよびwell missing-fraction metrics
- fixed LikPF/HMM 50:50 metrics
- well status manifest
- promotion gate JSON
- summary JSON

gzip predictionはraw gzip SHAとdecompressed content SHAを分け、
logical prediction content SHAを主証拠にする。

## 再現性設計

- seed policy: RNGなし。well、row、grid、rate、variant順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。保存LikPFはreadoutのみ。
- 並列処理: 実装時にouter worker / Numba threadを固定し、wellごとの出力順を
  sort keyで復元する。thread schedulingでRNGは変化しない。
- runtime: CPU、GPU/internet off。
- deterministic anchor: train-side scientific candidateの証拠としてSHAを記録するが、
  inference/submission未実装のためdeterministic submission anchorとは呼ばない。
- Kaggle package: prepare後にmetadata、loose config、package config、
  bootstrap展開configの一致を検証する。
- 記録: kernel id/version、input SHA、contract SHA、prediction raw/decompressed/logical
  SHA、metrics/gate SHAを記録する。model/submission SHAは対象外。

## リスク

- 科学リスク:
  heavy-tail化がwrong stateへの罰も弱め、absolute-TV​T mode識別を悪化させる可能性がある。
- tailリスク:
  平均RMSEが改善しても少数wellの誤mode固定が増える可能性があるため、
  p95/worst gateを必須にする。
- proxyリスク:
  exp342 Stage 0はfull HMMの符号を完全には予測しなかったため0-HMM先行gateを置かない。
- CV/LBリスク:
  exp209 branchはPublic LB anchorではない。train-side PASSでもinference/submissionは
  自動承認しない。
- runtimeリスク:
  773-well exact HMMはCPUで数時間かかる。control再実行を避け、candidate 1件に限定する。
- 再現性リスク:
  浮動小数並列順でraw gzipが変わり得るため、decompressed/logical SHAと
  metric tolerance`1e-5`を使う。
- multiple-testingリスク:
  `df=4`以外を試さず、結果後のdf/scale/temperature/grid救済を禁止する。

## 優先度

低・P4・CPU・terminal closed。ユーザーが意図したexp209単一変更を正しく
切り出す価値はあるが、exp342の小幅・tail不安定結果とexp209 emission関連の
複数negative resultを踏まえ、現行P1/P2を追い越さない。

## 実装反映

2026-07-24にcompact self-contained train/inference候補と専用contract testを実装し、
正規train Notebookへ採用した。Kaggle private CPU version 1で1 variant /
773 HMM well-runsを完了した。

## 次のアクション

directは`+0.217809 ft`、4/5 folds改善したが、by-well delta p95
`+0.982661 ft`とworst `+35.015963 ft`が固定tail gateをFAILした。
`student_t_exp209_failed_close_without_rescue`として再実行・救済・inference・
submissionなしで閉じる。
