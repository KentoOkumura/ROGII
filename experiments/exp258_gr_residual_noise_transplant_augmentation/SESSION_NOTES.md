# exp258 セッションノート

## 仮説

inner-trainだけに実測GR由来の連続残差blockを移植すれば、candidate pathや正解ラベルを変えずに
rankerの観測ノイズ耐性が上がり、そのrank-slot特徴で後段TVT LightGBMも改善すると仮定した。

## 2026-07-15 実装

ユーザー指示により、`gr_residual_noise_transplant_augmentation` を ranker 再学習で終わらせず、
後段の TVT 予測 LightGBM 再学習まで含む exp258 として実装した。

### 固定した親契約

- 親: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- route: `ml_model`。PF/Beam/HMM candidateはadd-only meta featureの補助利用。
- candidate: exp237 runtimeと同じ11本。
- selector: outer 5 × inner 4、candidate absolute-error regression 1 config。
- final: exp218 380 base features + exp238 35 rank-slot features、3 configs × 5 folds。
- parent/control再学習: なし。historical exp238 OOF artifactをbaselineとして読む。

### 実装内容

- `src/gr_residual_noise_augmentation.py` にrobust affine residual抽出、残差統計、stable SHA seed、
  continuous block/missing-mask移植、clean/white/shuffled controlsを実装。
- Stage 0 notebookで773 well相当のprofile inventory、affine誤差、missing run、FFT帯域、Haar DWT
  detail energy、profile/content SHA、20 nested splitのdonor/validation非重複を監査する。
- Stage 1 notebookではinner-train clean rowの一部を複製し、row IDを含むimmutable keyでsynthetic
  GR viewを作る。candidate pathは固定し、base 5 candidateのtarget-free `multiobs_*`列だけ再計算する。
- inner-valid / outer-validはaugmentationせず、historical exp238と同一foldでcalibration/safetyを比較する。
- Stage 2 notebookはselector summaryのprimary variant、20 models、guard pass、user approvalをhard
  assertし、3 × 5 = 15 GPU boostersを学習する。exp238 OOFとのbucket/fold/hidden-like/by-well guardを保存する。
- inference notebookはtrain guard通過まで停止し、推論時augmentationと再学習を禁止する。

### Kaggle train前コスト

- 現在の正規stage: `residual_audit_only`、0 boosters。
- primary selector train: `real_residual_block` 1 variant、1 config、outer 5、inner 4、20 CPU boosters。
- optional controls: `clean_duplicate` / `white_noise` / `shuffled_residual`を含む全4 variantなら80 CPU boosters。
- primary final train: real residual selector 1 variant、3 configs、5 folds、15 GPU boosters。
- exp238 baseline、exp218 parent、controlの再学習: なし。
- Kaggle push: 2026-07-15時点では未承認・未実行。

## 2026-07-16 ユーザー実行承認

- ユーザーが「実行してください」と明示した。
- 承認対象: primary `real_residual_block` 1 variant、candidate-error ranker 1 config、
  outer 5 × inner 4 = 20 CPU boosters。
- selector notebookは学習前に同じrunでStage 0 residual/fold-isolation auditを実行する。
- selector guard通過時だけ、primary 1 variant、final TVT LightGBM 3 configs × 5 folds =
  15 GPU boostersへ進む。
- negative controls 3種、historical exp238、exp218 parent/controlの再学習は行わない。
- inference / submissionは承認範囲外で、final OOF guard通過後も自動実行しない。

### Selector train v1 push

- Kernel: `kentookumura/exp258-gr-residual-noise-transplant-selector-train` v1
- Kaggle id_no: `127392032`
- Runtime: CPU、internet disabled、run_on_push=true
- 実行内容: Stage 0 residual audit後、primary `real_residual_block` 1 config、outer 5 ×
  inner 4 = 20 selector boosters。clean inner-valid / outer-valid、parent/control再学習なし。
- push後のpull metadataでcanonical id、CPU、internet off、competition source、9 kernel sourcesを確認。
- 現在の状態: 実行中。CLI logsが空でも同kernelへ再pushしない。
- ユーザー指示により、ローカルの`kaggle kernels logs -f`監視は停止した。Kaggle側のv1実行は
  cancelしていない。完了連絡を受けるまでstatus/log/output確認、再push、conditional final GPU
  trainを行わない。

### Selector train v1完了・guard不通過

- Kaggle status: `COMPLETE`
- Runtime: ログ最大時刻16,612.296秒（約4時間36分52秒）
- 入力: 3,783,989 rows / 773 wells / 11 candidates / 184 context features
- 実行: primary `real_residual_block`、outer 5 × inner 4の20 CPU boosters。best iterationは
  376..1200、outer/inner 20組を完全被覆。validation augmentationは全modelでfalse。
- residual audit: 773 wells / 5,092,255 raw rows、20 fold-isolation contract、donor/validation
  overlap最大0。
- historical exp238比:
  - global delta RMSE vs likpf: `-3.089911 -> -3.085357`（+0.004554悪化）
  - near 000_050: `-0.609540 -> -0.607888`（+0.001652悪化）
  - 1000+: `-3.372225 -> -3.365354`（+0.006871悪化）
  - worst-well regression: `+37.680897 -> +38.002960`（+0.322063悪化）
  - expected-error MAE: `4.532978 -> 4.523354`（-0.009625改善）
  - candidate AUC within 10 ft: `0.919334 -> 0.919159`（-0.000175悪化）
- guard: 6項目中expected-error MAEだけpass。global / near / longtail / AUC / worst-wellはfail。
- Decision: `selector_guard_failed_final_train_forbidden`。承認条件どおりfinal TVT LightGBM
  3 configs × 5 folds = 15 GPU boosters、inference、submissionは実行しない。
- 限定取得: residual audit summary、selector calibration / safety、20-model manifest、selector summary。
  5 nested score本体とmodel本体はダウンロードしていない。
- SHA256:
  - selector summary: `9e6575577ace80054073d28281885c80a3ed6b266116b400df6ffcafcbcae2b7`
  - selector model manifest: `8effd9232b895dd776ae29f85be1170224f98cf0b91f8d1786649dc9e7489fb4`
  - residual audit: `83668b9b0a4b18653eb8d11c6e21eba015a0411d518b6f114862e8426c71ae3d`
  - augmentation inventory decompressed: `630dac60d16f87ce08bb75289730e1f7544f29610b3d1fcdcad381afa9455af1`

### 再現性

- global RNGを使わず、seed / variant / outer / inner / recipient well / row ID / view slotを
  SHA256でuint64へ変換する。
- donor profile、fold manifest、row-level block contract、decompressed gzip score、model、OOFのSHAを分離保存する。
- selectorはCPU。finalは`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、8 threadsを継承する。
- Kaggle package bootstrapのconfig、stage、variant、GPU/internet、dependency sourceをpush前に検証する。

### 静的検証

- Jupytext変換 / `--to ipynb --test`: pass
- `py_compile`: pass
- `ruff --select F821,F401`: pass
- residual utility pytest: 4件pass
- strict `validate-exp`: pass
- selector / final停止状態package: config SHA一致、exp237 / exp238 / `src` bootstrap含有、
  exp238 historical OOF / exp115 hidden-like source含有、CPU/GPU metadata、internet off、
  `run_on_push=false`、user approval falseを確認。
- ローカル notebook 実行は行っていない。初回実行はKaggleを正とする。

## 次アクション

なし。selector guard不通過をもってこの分岐を終了し、final 15 GPU boosters、negative controls、
augmentation比率grid、inference、submissionは実行しない。
