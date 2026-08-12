# 要件

## 依頼

exp209 exact HMMのrate履歴による追従遅れを切り分けるため、TVTだけを持続状態とする
`41-rate memoryless`と、その特殊ケースである`dz-only (r_U=0)`を同一実験で比較する。

初回設計依頼の範囲は、バックログ、steering、design-only実験ディレクトリまで
だった。2026-07-29のユーザー追加指示`exp435を実装してください`により、
compact self-contained train、contract test、fail-closed inference、
正規Notebook採用までを今回の追加範囲とする。

その後の2026-07-29のユーザー指示`実行してください。`により、固定済みの
Kaggle package / push / Stage 0 runだけを追加承認範囲とする。Stage 1、
inference実行、submissionは引き続き別承認とし、今回行わない。

## 実験の状態定義

- 現行exp209:
  持続状態は`(TVT, r_U)`。`r_U=dU/dMD`の前行posteriorを次行へ伝える。
- `memoryless_41rate`:
  持続状態はTVT上の確率分布だけ。各行で41個の`r_U`候補を固定重みで周辺化し、
  その行のrate responsibilityを次行へ伝えない。
- `dz_only_r0`:
  持続状態はTVT上の確率分布だけ。`r_U=0`に固定する。
- 両treatmentとも、TVT posterior mean 1点ではなくTVT確率分布全体を次行へ伝える。
- `U=TVT+Z`、`ΔTVT=r_U ΔMD-ΔZ`と定義する。

## 制約

- 対象実験は`exp435_tvt_memoryless_u_rate_dzonly_hmm`。
- Routeは`pf_beam`。
- 科学的親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 原因証拠は`exp408_hmm_message_rate_basin_audit`。
- 補助証拠として、momentum単独介入が失敗したexp424と、非ゼロgeometry rateに
  平均signalがあったがhidden-like / tailを壊したexp355を参照する。
- 2 treatmentを同じStage 0で比較し、一方だけを先に選んで実行しない。
- 親exp209の保存済みOOFを対照とし、親HMMを再実行しない。
- TVT grid、初期TVT prior、Type Well GR emission、GR preprocessing、
  position noise、forward-backward smoothing、posterior-mean readoutを親から固定する。
- `memoryless_41rate`のrate gridは親と同じ41点のzero-centered supportとする。
- 毎行のrate重みは、親の`mom=0.998`、`sig_r=0.002`から導く
  zero-centered stationary Gaussianをgrid上で正規化した固定重みとする。
- prefix `init_rate`は親互換のsupport幅を決める用途だけに使い、rate重みの平均、
  前行posterior、row-adaptive重みには使わない。
- `dz_only_r0`は`memoryless_41rate`と同じposition-only kernelへ
  `rates=[0]`, `weights=[1]`を渡す特殊ケースとして定義する。
- Stage 0はexp411のSHA固定済みfixed32
  （persistent 16 / matched control 16）をmechanism-only sampleとして使う。
- Stage 0実行量は2 treatment × 32 wells = 64 HMM well-runs。
- Stage 1はStage 0を通過したtreatmentだけを各773 wellsで評価し、
  最大2 treatment × 773 wells = 1,546 HMM well-runsとする。
- model、LightGBM config、trained fold、booster、PF、Beam、GPUは0。
- Stage 0、Stage 1、inference、submissionはそれぞれ別の明示承認なしに行わない。
- suffix truth、fold、hidden-like role、episode、errorは全candidate predictionと
  readoutのfreeze後にだけjoinする。
- 同一OOFを見たrate重み、rate support、GR sigma、position sigma、grid、
  gate、blend weight、well / row selectorの救済を行わない。
- 再現性は`docs/06_reproducibility.md`に従い、入力、manifest、prediction、
  transition contract、metricsのcontent SHAを記録する。

## 受け入れ基準

- 現行HMM、41-rate memoryless、dz-onlyの状態と因果比較を一意に記述する。
- `memoryless_41rate`のrate gridと固定重みを数式と設定値で固定する。
- `dz_only_r0`が同一kernelのdelta-rate特殊ケースであることを固定する。
- Stage 0 / Stage 1の実行量、technical gate、mechanism gate、promotion gateを固定する。
- fixed32がmechanism-onlyであり、full OOFだけがpromotion判断になると明記する。
- dz-onlyは新しいabsolute TVT anchorを追加しないことを明記する。
- direct HMM評価に加え、exp263固定式のHMM成分だけを置換するadoption readoutを固定する。
- config、README、SESSION_NOTES、result、metricsをdesign-only状態で作る。
- `KAGGLE_DIRECTION.md`へ、2 treatmentを含む1件のdesign-only backlogとして登録する。
- `experiment_summary.md`へdesign-only実験を記録する。
- 実装と正規Notebook採用を完了する。
- Kaggle package / push / Stage 0 runを固定契約どおり完了し、gateを記録する。
- Stage 1、inference実行、submissionは今回の依頼に含めない。
