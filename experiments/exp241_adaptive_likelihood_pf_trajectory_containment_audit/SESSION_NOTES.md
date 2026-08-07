# exp241_adaptive_likelihood_pf_trajectory_containment_audit セッションノート

## 目的

target-free gate 後に PF path が resampling / seed aggregate で長期発散するかを診断する。

## 現在の状態

- 状態: 完了・部分train-side監査で不採用、shard 1未実行、inference/submitなし
- route: `pf_beam`
- 親: `exp232_adaptive_robust_likelihood_pf`
- inference / submission: 無効

## 実行予定コスト

- active treatment: 1 (`gated_t2`)
- PF control replay: 1 (`paired_t1_control`)
- PF treatment replay: 1
- 合計 PF replay: 2
- well shard: 4固定、各kernelで1 shard（約193 wells）
- particles: 500
- seeds: 128
- LightGBM config: 0
- folds: 0
- boosters: 0
- GPU: なし（Kaggle CPU）
- parent/control model再学習: なし。ただし instrumented T=1 PF control を診断用に再生成する。

exp232 の単一 replay が約 9.5～10.9 時間だったため、full 773 wellsでの直列2 replayは
12時間上限を超える。stable SHA256 well moduloで4 shardへ分割し、各shardでcontrol+treatmentを
実行する。4 shardのactive indexとpackageをpush前に再確認し、自動pushはしない。

## 実装内容

- T=1/T=2 を同一 well seed base で paired replayする。
- seed-level ESS、gate、resampling matrix はwell処理中だけ保持し、event summaryへ縮約する。
- event は各 well×seed の最初の treatment gate とし、targetはevent選択後のscoreだけに使う。
- horizon `8/32/64/128/256/512/1024/end` で cumulative RMSE delta、path divergence、
  ESS、resampling countを出す。
- row-level seed std/p10/p90、control/treatment meanを保存する。
- conditional resampling後の乱数消費分岐は隠さず、最初のresampling divergence rowを記録する。

## コマンドログ

```bash
make new-steering EXP=exp241_adaptive_likelihood_pf_trajectory_containment_audit
make new-exp EXP=exp241_adaptive_likelihood_pf_trajectory_containment_audit SOURCE=experiments/exp232_adaptive_robust_likelihood_pf
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp241_adaptive_likelihood_pf_trajectory_containment_audit/exp241_adaptive_likelihood_pf_trajectory_containment_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp241_adaptive_likelihood_pf_trajectory_containment_audit/exp241_adaptive_likelihood_pf_trajectory_containment_audit_train.py
.venv/bin/python -m py_compile experiments/exp241_adaptive_likelihood_pf_trajectory_containment_audit/*.py
.venv/bin/ruff check experiments/exp241_adaptive_likelihood_pf_trajectory_containment_audit --select F821
make validate-exp EXP=exp241_adaptive_likelihood_pf_trajectory_containment_audit
make update-summary
```

- Jupytext train/inference test: pass。
- `py_compile`: pass。
- Ruff F821: pass。
- strict experiment validation: pass。
- Numba runtime smokeはローカル環境に`numba`がなく未実行。Kaggle packageは親exp232と
  同じNumba前提だが、初回実行時に小規模compile cellの完了を確認する。
- 親exp232正規train notebookは246行/6章、exp241は173行/6章。exp241 notebook上で
  input、paired replay contract、event policy、cost、実行、metrics/生成物を追える。

## 再現性

- `docs/06_reproducibility.md` を確認済み。
- stable seed: SHA256(exp241, well, paired namespace) + seed index。
- Numba single worker、thread parallel RNGなし。
- gzip生成物はdecompressed content SHAを主証拠としてmetricsに記録する。
- stochastic PF監査であり deterministic submission anchor とは扱わない。

## 次のアクション

なし。2026-07-13のユーザー判断によりshard 1は実行せず、574/773 wellsの部分監査で
exp241を閉じる。temperature/mixture/process-noise再grid、raw-test inference、submitへ進まない。

## 暫定結果（shard 0/2/3）

- 集計範囲は574/773 wells（74.3%）、2,813,393 rows。shard 1未実行のため最終判定ではない。
- gated T=2はpaired regenerated T=1に対し、全3 shardで僅かに改善した。pooled RMSEは
  `13.223203` vs `13.235174`（delta `-0.011971`）、`1000_plus`も`14.432237`
  vs `14.444841`（delta `-0.012604`）。
- 一方、保存済みexp072 `likpf_mean`は`11.323679`で、regenerated T=1は`+1.911495`、
  T=2は`+1.899524`悪い。exp232で見えた約+1.93の悪化の大半はgate treatmentではなく、
  paired replayと保存済みexp072 predictionの再現差に由来する。
- hidden-likeではT=2がpaired T=1より僅かに悪化した。spatialは`+0.012390`、
  typewell-purgedは`+0.011157`。worst-well deltaの最大は`+1.666805`でguard不通過。
- first-gate event後のmean absolute path divergenceは8 rowsで`0.127775 ft`、1024 rowsで
  `1.399960 ft`、endで`3.123176 ft`へ増加した。terminal absolute divergenceは平均
  `5.430025 ft`、event内maxは平均`10.475701 ft`で、trajectory containmentは支持されない。
- ただしevent単位のend cumulative RMSE deltaは平均`-0.016838`、正のdelta率`49.66%`で、
  発散が一方向の誤差悪化を生む証拠はない。T=2は短期ESSがやや高く、resamplingも少ない。
- guardはoverall/1000_plusのみ通過し、hidden-like、worst-well、late divergenceは不通過。
  全4 shardの厳密な母集団推定ではなく部分監査だが、direct robust likelihoodを不採用とする
  証拠として十分と判断し、ユーザー承認のもと完了扱いにする。

## Kaggle実行

- 2026-07-13: 4 packageのmetadata/bootstrapを確認。各shardはcontrol 1、treatment 1、
  500 particles、128 seeds、LightGBM 0、fold 0、booster 0、CPU、internet disabled。
- shard 0: `kentookumura/exp241-containment-audit-shard0` v1 push成功、run開始。
- shard 1: canonical pushは`Maximum batch CPU session count of 5 reached`で拒否。その後
  canonical idはpush時`Notebook not found`、pull 500、status 404でresource未作成を確認。
  recovery slug `kentookumura/exp241-containment-audit-shard1-retry`を同一設定でprepareしたが、
  再pushもCPU session上限5で拒否。slot待ちでありrun未開始。
- shard 2: `kentookumura/exp241-containment-audit-shard2` v1 push成功、run開始。
- shard 3: `kentookumura/exp241-containment-audit-shard3` v1 push成功、run開始。
- shard 0 v1完了: 180 wells、生成物保存完了、notebook elapsed約18,372秒。
- shard 2 v1完了: 210 wells、生成物保存完了、notebook elapsed約17,142秒。
- shard 3 v1完了: 184 wells、生成物保存完了、notebook elapsed約18,975秒。
- 2026-07-13: shard 1 retry / retry2 は前回slot拒否後status 404を確認。slot解放待ち後に
  retry3 packageまで再prepareしたが、Kaggleは引き続き`Maximum batch CPU session count of 5 reached`
  を返した。科学設定は不変、shard 1は未開始。
- 2026-07-13: ユーザーからexp241を閉じてよいとの判断を受けた。shard 1 retryは中止し、
  shard 0/2/3の574 wells / 2,813,393 rowsを最終的な部分監査範囲として記録した。
