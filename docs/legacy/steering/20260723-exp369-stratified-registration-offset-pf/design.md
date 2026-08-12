# 設計

## アプローチ

exp072粒子を`(p,r,delta)`へ拡張する。deltaは`[-6,-3,0,3,6] ft`、初期500粒子を
`[50,75,250,75,50]`に固定する。各行で隣接deltaへ方向ごとに`1/512`で遷移する。
resampling時は各delta層を最低25粒子残し、残り375をposterior delta massでsystematic配分する。
GR emissionは`p+delta`、TVT出力は`p`。

Stage 0はvisible prefixのみで128行history / 64行held-out / stride64のrolling-originを行う。
delta=0比のheld-out GR predictive NLL、circular control、nonzero/boundary mass、隣接windowの
delta符号安定性をAND評価する。exp365とは値を共有せず独立に判定する。

Stage 1は全gateと別承認時だけ500 × 128 × 773の1 treatment PFを実行する。controlは保存済み
exp072 likpf_mean。失敗時にdelta、quota、transition、particle/seed、affine/DTWを調整しない。

## 実験範囲

- 対象: `exp369_stratified_registration_offset_pf`
- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- 変更: bounded delta stateとdelta-stratified resamplingだけ。
- 固定: physical dynamics、500 particles、128 seeds、likelihood sigma、ESS/resampling、mean aggregation。
- Stage 0 gate: NLL gain`>=1%`、4/5 folds、circular差`>=0.5%`、nonzero mass`[0.05,0.50]`、
  boundary mass`<=0.25`、隣接window符号一致`>=0.60`。
- Stage 1 gate: exp072比`>=0.05 ft`、4/5 folds、1000+/hidden-like/p95回帰`<=0.02 ft`、
  worst`<=0.25 ft`、各delta posterior mass`>=0.01`。

## 再現性設計

- seed: `SHA256(experiment|well|family|seed_index)`からlocal RNG。
- global RNG、thread schedule依存、stream共有は禁止。
- stochastic: initial jitter、delta transition、propagation、resampling、jitter。
- CPU single worker、GPU off、上限30,600秒。
- raw train/testを別生成し、window/delta diagnostics/predictionのcontent SHAを保存する。
- gzipはdecompressed SHA。Stage 0はsuffix truthを読まない。

## リスク

- leakage: held-out GRをhistory posteriorへ混ぜる危険。window境界を固定する。
- CV/LB不一致: registration分布差。
- runtime: 500粒子固定だがstratified管理が増える。
- reproducibility: quota丸めとdelta orderを固定する。
- science: delta=0のparticle解像度低下とwrong-mode維持。
