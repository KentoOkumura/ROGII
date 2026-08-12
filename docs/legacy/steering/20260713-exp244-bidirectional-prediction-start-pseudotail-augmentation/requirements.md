# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `bidirectional_prediction_start_pseudotail_augmentation`
を `exp244_bidirectional_prediction_start_pseudotail_augmentation` として実装する。
exp239 の early-start replay contract を拡張し、official prediction start の前・同位置・
後ろに固定少数の start を置いた `early` / `original` / `late` multi-view pseudo-tailを作る。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- late-startでofficial startより後ろに追加するtrue TVTはtrain-only augmentationに限定する。
- current-testで再現可能なrolling prefix calibrationは別契約として分離し、actual prediction startを越えない。
- 同じsource wellの派生viewは同一foldを継承し、outer-valid source well由来viewをouter-trainへ入れない。
- full-prefix生成済みcacheのslice流用は禁止し、startごとにprefix依存特徴を再生成する。
- 初版はCPU deterministic auditとし、LightGBM学習、親/control再学習、推論、提出、Kaggle pushを行わない。
- MTP/CNNは対象外とする。

## 受け入れ基準

- raw trainのofficial cutoffを基準に、configで指定した固定row offsetからearly/original/late view manifestを決定的に生成できる。
- 各viewにstart kind、officialとの差、prefix rows、remaining tail rows、source well、fold、target usageを保存する。
- early/original/late比率、残tail長、well coverage、推定augmentation rowsを監査し、config capを検証する。
- 各startで`TVT_input`を再構成し、anchor/prefix統計をfull-prefix cache非依存でmaterializeできる。
- PF/Beam、learned likelihood、GR confidence、PF/HMM初期状態を後続再生成するreplay contractを保存する。
- official-start OOFを主評価に固定し、start方向、残tail長、1000+、hidden-like、worst-wellを後続評価面として記録する。
- source-well fold alignment、outer-valid除外契約、late-start train-only、test unknown-tail非参照をhard assertionで検証する。
- Jupytext percent形式のself-contained train/inference notebookとipynbが静的検証を通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## v2 frozen-anchor parity preflight

- exp218 train v1 の保存済み `lgb_mean` OOF と model manifest を読み、experiment / variant / mode / model、行数、well数、RMSE、content SHAを照合する。
- raw train のofficial tail行数とexp218 OOFのwell別行数・ID範囲が完全一致することをhard assertionで確認する。
- exp218が行レベル `GroupKFold` で作ったfoldをofficial tail行数から再構成し、v1のwell均等foldとの差を記録する。
- v2以降のsource-well foldはexp218互換foldへ固定し、v1 foldをexp218との性能比較に使わない。
- parity preflightはCPU、0 LightGBM config、0 fold学習、0 booster、親/control再学習なしで実行する。
- parityが通るまでcalibrator学習、予測、提出を行わない。

## v3 dual-start confidence-shrink meta-validation

- v2 parity fold manifestとexp218 frozen `lgb_mean` OOFをidentity SHA付きで入力する。
- 各wellのactual/official start以前だけを使い、`-1000`と`-250`の2 startから固定local-linear backtest errorを計測する。
- shrink式は事前固定のsingle variantとし、official-tail truthや他wellのtruthからparameterをfitしない。
- 2 startの両方でbacktest RMSEが10 ftを超える場合だけ、10〜30 ftの固定rampでexp218 residualを最大5% anchorへ縮める。
- `pred = anchor + alpha * (exp218_pred - anchor)`、`alpha in [0.95, 1.00]`をhard assertionする。
- primaryはv2 fold上のofficial-start OOF。overall、fold別、距離bucket、1000+、exp115 hidden-like、by-well、worst-well、使用率、start間安定性を保存する。
- overall、1000+、hidden-like 2面が非悪化、worst-well回帰+2 ft以内、5-fold中3-fold以上改善を採用guardとする。
- full model fine-tune、OOF truthを使うalpha fitting、per-well grid、test prediction、submissionを禁止する。
- CPU、active variant 1、LightGBM config 0、fold学習0、booster 0、親/control再学習なしで実行する。

## v4 early / original / late 統合学習

- v3 calibratorは本実験の中心仮説を検証していないため、early / original / lateを実際の
  exp218-family学習データへ入れるsingle variantを実装する。
- originalは保存済みexp239 official 380-feature cacheの全3,783,989行を使う。original viewを
  pseudo cacheへ重複追加しない。
- pseudoは`-1000/-250/+250/+1000`の全成立viewを使い、各viewを固定5距離帯から各50行、
  最大250行へ決定的にsampleする。期待値は3,081 views / 770,157 rowsとする。
- pseudo cacheはoffset別の4 CPU notebookへ分ける。各cacheは同じ380-feature schema、request/source
  well、row/content/file SHAを保存し、単一12時間jobへ全requestを詰めない。
- 学習はofficial weight 1.0、pseudo weight 0.5のsingle variantとする。active variant 1、
  LightGBM config 3、fold 5、合計booster 15、親exp218/control再学習なしとする。
- outer-valid source well由来のearly/late rowを対応foldのtrainから除外し、validationは
  official-start全行だけに固定する。
- 保存済みexp218 OOFをcontrolとし、overall、fold、距離bucket、1000+、hidden-like 2面、
  by-well、worst-wellを同じofficial surfaceで比較する。
- overall改善、1000+非悪化、hidden-like 2面非悪化、worst-well回帰+2 ft以内、5 fold中3 fold以上
  改善を採用guardとする。
- feature cache生成はCPU / 0 booster。GPU trainは4 cacheのSHA契約通過後だけ実行し、push前に
  1 variant / 3 configs / 5 folds / 15 boostersをユーザーへ提示して明示承認を得る。
- inference、current-test prediction、submissionはtrain-side guard通過まで禁止する。
