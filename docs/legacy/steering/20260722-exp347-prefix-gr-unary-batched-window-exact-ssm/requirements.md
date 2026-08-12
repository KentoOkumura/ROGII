# 要件

## 依頼

exp332の「固定長window内でstructured objectiveを学習する」という着眼点を維持し、複数windowを同時にGPUへ載せるbatched exact DP案を、バックログ・steering・実験ディレクトリへdesign-onlyで固定する。実装、正規Notebook編集、Kaggle package/push/run、推論、提出は行わない。

## 2026-07-22 実装承認追記

ユーザーの明示依頼「exp347を実装してください」により、別名compact self-contained train/inference候補、batched exact DP、専用contract tests、設定・記録の実装を承認範囲へ追加する。正規Notebook上書き、Kaggle package/push/run、Stage A/B/C、推論、提出は引き続き未承認とする。

## 2026-07-22 Stage 0実行承認追記

ユーザーの明示依頼「実行してください」により、compact self-contained train候補の正規train Notebook採用と、固定16-window Kaggle T4 Stage 0のpackage/push/runを承認範囲へ追加する。実行量はbenchmark variant 1、一時neural model 1、永続model/fold/LightGBM/booster/PF/Beam/control/親再学習0で固定する。Stage A/B/C、正規inference採用、推論、提出は引き続き未承認とする。

## 仮説

exp332の1 window/batch + gradient accumulation 4を、4 windows/batch + accumulation 1へ等価にまとめ、windowごとのnormal / label-conditioned exact forward-backwardをbatch次元で並列化すれば、Gaussian soft-label structured objective、optimizerの実効batch、full-well評価契約を変えずに、保守的fold runtimeを`13.151137 h`から`8.5 h`以下へ短縮できる。

## 制約

- Route: `ensemble`。
- 親実験はterminal closedの`exp332_prefix_gr_unary_fixed_window_structured_ssm`とし、exp332自体は再開・変更しない。
- 科学契約はexp332を固定する。256 rows、3 scheduled slots、最大3 active windows/well/epoch、8 epochs、structured NLL重み`1.0`、Gaussian sigma`0.35 ft`、local CE重み`0.25`、architecture、teacher boundary、41-rate exact SSM、controls、full-well decodeを変えない。
- 唯一の変更は計算実装。連続する4 window lossの平均を1 batchで計算し、exp332の4回gradient accumulationと同じ1 optimizer updateに対応させる。
- paddingされたposition/rate/state cellは`-inf` potentialと明示maskでloss、posterior、gradientから除外する。
- Stage 0ではscalar exp332実装とのforward loss、posterior、gradient、1-step parameter update parityを確認する。
- Stage 0実行は1 benchmark variant、固定16 windows、一時neural model 1、永続model 0、LightGBM config/booster/PF/Beam/control再学習0。別承認なしに実装・GPU実行しない。
- Stage AはStage 0全gate PASSと別承認後だけ、fold 0 / architecture 1 / seed 42 / neural model 1。Stage B/C、推論、提出はさらに別gate・別承認とする。
- 再現性は`docs/06_reproducibility.md`に従い、window/batch/boundary manifest、入力、model、prediction、package/kernelのSHAを記録する。
- batch size、padding bucket、window数/長、loss、decoder、architecture、epochのgrid探索を同じ実験で行わない。

## 受け入れ基準

- `docs/legacy/steering/20260722-exp347-prefix-gr-unary-batched-window-exact-ssm/`と`experiments/exp347_prefix_gr_unary_batched_window_exact_ssm/`が存在し、仮説、単一変更、段階gate、禁止事項、再現性、実行量が記録されている。
- `config.yaml`は`design_only_not_implemented`、`implementation_approved=false`、Kaggle/inference/submission承認falseでfail-closedになっている。
- Stage 0 technical parityは、loss/posterior max abs error`<=1e-6`、gradient/1-step update max abs error`<=1e-5`、finite率1.0を必須とする。
- Stage 0 compute gateはT4の保守的fold外挿`<=8.5 h`、peak GPU memory`<=14 GB`、exp332比speedup`>=1.55x`をすべて要求する。
- Stage 0 FAIL時はbatch sizeやscience contractを救済せずbranchを閉じる。
- 実装コード、compact self-contained source、正規Notebook採用、Kaggle package/output、学習済みmodel、prediction、submissionがまだ存在しないことを明記する。
- deterministic anchorとは扱わない。将来gzip生成物を比較する場合はdecompressed content SHAを主証拠にする。
