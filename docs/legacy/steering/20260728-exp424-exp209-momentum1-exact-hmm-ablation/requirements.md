# 要件

## 依頼

exp408で確認されたexact HMMのrate絶対値の0方向under-responseに対し、
exp209の`mom=0.998`だけを`1.0`へ変更する独立ablationを設計確定する。
バックログ、steering、実験ディレクトリまでを作成し、実装、Notebook編集、
Kaggle package / push / run、inference、submissionはまだ行わない。

## 制約

- Routeは`pf_beam`。
- 科学的親は
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、
  原因証拠は`exp408_hmm_message_rate_basin_audit`とする。
- 変更する値はHMM rate transitionの`mom: 0.998 -> 1.0`だけ。
- `sig_r=0.002`、rate grid、position transition、GR preprocessing / emission、
  initial prior、forward-backward、posterior-mean readoutは固定する。
- `mom=1.0`は0方向mean reversionを除くが、rate noiseやstate supportは増やさない。
- exp338で悪化した`sig_r=0.004`、exp411のdirectional trigger、
  exp412のtwo-pass beta scheduleとは混ぜない。
- Stage 0はexp411のSHA固定済みfixed32
  （persistent 16 / matched control 16）を診断sampleとして再利用する。
- Stage 0ではsample-matchedなfiltered / smoothed rate moment比較が必要なため、
  baseline 32 + treatment 32 = 64 HMM well-runsを実行する設計とする。
- 保存済みexp209 predictionはbaseline passのTVT parity確認にも使う。
- Stage 0 sampleはpersistent-error情報を使って選ばれているため、
  promotion根拠には使わず、mechanism preflightに限定する。
- Stage 1はStage 0全gate PASSと別承認後だけ、1 treatment / 773 HMM well-runsを
  full OOFで評価する。full OOFの親controlは保存済みexp209 predictionを使い、
  再実行しない。
- parent control再実行を含むStage 0と、高コストなStage 1の各Kaggle実行前に
  明示承認を得る。
- model、LightGBM config、trained fold、booster、PF、Beam、GPUは0。
- 同一OOFでmomentum、`sig_r`、rate grid、position noise、emission weightを探索しない。
- 再現性は`docs/06_reproducibility.md`に従い、入力、sample manifest、
  prediction、rate readout、metricsのcontent SHAを記録する。

## 受け入れ基準

- `mom=1.0`が実装式上どのmean driftを除くかを一意に記述する。
- Stage 0 / Stage 1の対象、実行量、technical / scientific gateを固定する。
- truth / episode情報をHMM入力へ渡さず、prediction freeze後にだけ評価へjoinする。
- fixed32がmechanism-onlyであり、full OOFだけがpromotion判断になると明記する。
- exp338 / exp408 / exp411の既存結果と矛盾しない禁止事項を定める。
- config、README、SESSION_NOTES、result、metricsをdesign-only状態で作る。
- `KAGGLE_DIRECTION.md`の既存候補と比較してP3へ配置する。
- 実装・Kaggle実行・inference・submissionは今回の依頼に含めない。

## 2026-07-28 追加依頼: 実装

ユーザーの追加依頼「exp424を実装してください」により、上記design-only境界のうち
実装と正規Notebook採用だけを解除する。今回追加で受け入れる範囲は次とする。

- compact self-contained Jupytext train sourceと正規train Notebookを実装する。
- Stage 0のparent 32 + treatment 32を実行できるが、未承認時はfail-closeする。
- parent / treatment prediction、filtered / smoothed rate moment、rate edge massを
  truth-late境界の前にfreezeし、content SHAを保存する。
- exp209 small-trellis parity、`mom=1.0` transition mean、fixed32 SHA、
  truth-late、inference禁止を専用testで固定する。
- fail-closed inference placeholderと正規inference Notebookを実装する。

解除しない範囲:

- Kaggle package / push / run。
- Stage 0結果の生成。
- Stage 1の実装・実行。
- inference prediction、submission。

## 2026-07-28 追加依頼: Stage 0実行

ユーザーの追加依頼「実行してください」により、事前登録済みStage 0の
Kaggle CPU package / push / runだけを解除する。

- baseline 32 + treatment 32 = 64 HMM well-runsを実行する。
- model / booster / PF / Beam / GPUは0を維持する。
- fixed32はmechanism-onlyであり、CV / promotion evidenceとは呼ばない。
- canonical kernel idを使い、別slugを作らない。
- logs / Notebook cell outputからgate、runtime、SHAを記録する。

Stage 1、inference、submissionは解除しない。

## 2026-07-28 Stage 0結果

canonical Kaggle private CPU kernel Version 1でbaseline 32 + treatment 32の
64 HMM well-runsを完走した。

- technical gate: 13 / 13 PASS
- mechanism gate: 3 / 7 PASS
- persistent episode SSE reduction:
  `0.475550% < 5%`
- persistent improved wells:
  `8 / 16 < 10 / 16`
- persistent improving folds:
  `3 / 5 < 4 / 5`
- under-response SSE share reduction:
  `9.849995 points >= 2 points`
- control pooled / by-well p95:
  `-0.054769 / +0.157066 ft`、ともにPASS
- smoothed rate edge mass delta:
  `+0.000377954`、nonworse FAIL

事前固定fail policyに従い`stage0_fail_closed`とする。Stage 1資格はなく、
inference / submissionも実行しない。
