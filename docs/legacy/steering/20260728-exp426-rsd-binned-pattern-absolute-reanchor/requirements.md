# 要件

## 依頼

Wu et al. (2019) の stochastic clustering / pattern matching を、
exp408 で確認された HMM の translation-gauge lock と、
exp226 で確認された suffix 累積 offset に対する absolute datum 再アンカーへ
転用できるかを段階的に検証する。

今回の依頼範囲は、バックログ、steering、実験ディレクトリと設計の確定までとする。
実装、正規 Notebook 編集、Kaggle package / push / run、inference、
submission はまだ行わない。

## 2026-07-28 追加依頼

ユーザーの「exp426を実装してください」をStage A実装の明示承認として扱う。
今回追加する範囲はcompact self-contained Jupytext source / notebook候補と
contract tests、implementation-ready記録までとする。

- 正規train / inference Notebookは上書きしない。
- Kaggle package / push / runは行わない。
- Stage B / Cは実装しない。
- inference / submissionは行わない。

## 制約

- Route は `pf_beam` とする。
- 論文:
  Wu et al., “Stochastic clustering and pattern matching for real-time
  geosteering,” Geophysics 84(5), 2019,
  DOI `10.1190/geo2018-0781.1`。
- 科学的な基準 path は
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
  の保存済み fold-safe OOF `tvt_pred` とする。
- HMM の原因証拠は `exp408_hmm_message_rate_basin_audit`、
  PF の原因証拠は `exp410_likpf_particle_resampling_basin_audit` とする。
- PF の科学的親は `exp404_scale5_sigma_gr_likelihood_pf_ablation` の
  x1.0 / temperature-5 candidate とする。
- exp226、exp404、exp209 の保存済み control は再生成しない。
- Stage A/B は model、booster、HMM、PF、Beam、GPU をすべて 0 とする。
- Stage C だけが likelihood-PF を実行できる。Stage A/B の全 gate PASS、
  設計レビュー、ユーザーの別承認を必要とする。
- Stage A は exp280/exp360 と同じ non-overlapping 512-row block と
  13 offset states
  `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft` を使う。
- primary score は論文どおり、candidate path を RSD 0.5 ft bin へ写像し、
  bin 内 raw finite GR の平均と Type Well GR の Pearson 相関を Fisher 変換した
  signed score とする。
- Pearson / offset grid / block / bin 幅 / support / transition penalty /
  PF anchor weight / PF anchor widthを、同じ OOF の結果を見て変更しない。
- Cosine と Spearman は論文再現の descriptive readout に限定し、
  primary の救済には使わない。
- 論文の dip / inclination prior は fixed-offset candidate 間で定数となるため、
  Stage A の candidate rankingへ重複加算しない。exp226 pathをlocal geometry prior
  として固定する。
- query truth、error、oracle offset、persistent episode、hidden-like role は、
  target-free score / candidate / prediction と content SHA の freeze 後だけ joinする。
- Stage C の pattern score は proposal allocation にだけ使い、
  `p/q` importance correctionにより raw GR likelihoodへ二重加算しない。
- Stage C は元 PF continuation を 90% 残し、全13 absolute-anchor componentへ
  nonzero supportを持たせる。top-3だけにsupportを切らない。
- 再現性は `docs/06_reproducibility.md` に従い、per-well stable seed、
  input / score / candidate / prediction / diagnostic の logical content SHA、
  Kaggle kernel versionを記録する。
- full PF は 500 particles ×128 seeds ×773 wells の高コスト処理なので、
  sentinel PASS後も別承認なしに実装・実行しない。

## 受け入れ基準

- Stage A が「absolute datum offsetをGRから識別できるか」、
  Stage B が「exp226を安全に再アンカーできるか」、
  Stage C が「PFのfinite supportとdatum basinを改善できるか」を別gateで判定する。
- Stage A/B/C の入力、式、実行量、technical / scientific gate、
  fail-close条件と禁止救済が一意に記録される。
- Stage A FAILならB/C、Stage B FAILならCへ進まない。
- Stage A は matched raw Pearson、raw Gaussian、stable permutationをcontrolに持ち、
  RSD binningの寄与を分離する。
- Stage B は exp226 のlocal shape/rateを変更せず、coarse datum correctionだけを
  連続に補間する。
- Stage C は同じ augmented PF targetに対するuniform-anchor proposalと
  pattern-guided proposalを比較し、物理reanchorと論文scoreの寄与を分離する。
- HMMについては、Stage Aをresidual-datum state導入の共通必要条件として扱い、
  exact HMM自体の改善をこの実験結果だけで主張しない。
- `config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、
  `metrics.json` を design-only 状態で作る。
- `KAGGLE_DIRECTION.md` の未着手バックログへ、既存negative evidenceを含む
  P3候補として追加する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、
  prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく
  decompressed content SHA を主証拠として記録している。
- 実装、Kaggle実行、inference、submissionは今回の依頼に含めない。
