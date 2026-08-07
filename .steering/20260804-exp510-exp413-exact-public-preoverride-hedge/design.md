# 設計

## アプローチ

公開notebookの高いPublic LBをそのまま信用せず、予測生成を次の3層に分ける。

1. dual-pipeline core: projected SP45とlearned Fleongg branchの固定blend。
2. guarded contact override: train/test overlap wellへのcontact再構成置換。
3. Gold overlay: test well自身のvisible prefixで候補を選ぶper-well calibration。

本実験は1だけをfreezeし、2と3を完全に除外する。さらにexp413へ10%だけ混ぜることで、
公開系trajectoryのPublic分布相補性を残しながら、Privateへ転移しない可能性のある補正量を抑える。
10%は最終提出portfolio用の事前固定値であり、exp497の13.716%係数をexact public componentへ
流用した値ではない。

## 実験範囲

- 対象実験: `exp510_exp413_exact_public_preoverride_hedge`
- Route: `ensemble`
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- public source: `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260627/degnonguidi__public-score-rogii-lb-7-159/public-score-rogii-lb-7-159.ipynb`
- archived source SHA256: `4d0712983788dc7d9b97fdb8e5dc7c30b6d3634a9c64597d84d21da28e9623eb`
- 変更する変数: public sourceのpre-override componentを10%追加する。
- 固定する変数:
  - `public_preoverride = 0.55 * sp45_projection_submission + 0.45 * submission_B`。
  - final=`0.90 * exp413 + 0.10 * public_preoverride`。
  - Gold OFF、guarded contact 0、同一well lookup 0。
  - source内のSP45 selector/projection、Pipeline B feature/model/postprocess構造。
- `submission_A.csv`へ置換しない。可視sourceのfinal cellが読むsidecarを正とする。
- 実装時inventory予定: scientific variant 1、final blend 1、new train model/config/fold/booster 0。PF/Beam run数はhidden well数から動的算出し、push前に記録する。
- 状態ラベル: `implementation_complete_candidate_not_adopted_public_distribution_hedge`。

## 実装時のartifact解決

- Runtime必須modelは`fleongg/rogii-claude-models-pub` version 1の`features.json`と
  `lgb0/1/2.pkl`だけと確定した。4ファイルはSHA固定する。
- projected-SP45がsample全IDをcoverしなければfail-closeするため、Pipeline-A tabular fallback、
  `koolbox`、`ravaghi`の7GB train feature/trainerはruntime routeから外す。
- archived metadataに列挙された残りdataset/kernel sourceはlineageとして記録するが、候補notebookへ
  mount/importしない。
- 正規notebookは上書きせず、`*_compact_selfcontained_inference.py/.ipynb`を候補として作る。

## source境界と禁止scan

- freeze point: final blend cellが`submission.csv`を書いた直後、`run_guarded_contact_override`呼び出し前。
- sourceから取り込むもの:
  - SP45 likelihood-PF/Beam selector。
  - robust degree-4 projection。
  - Pipeline B likelihood-PF、spatial priors、保存済みlearned model inference、warmup/SG。
  - 0.55/0.45 component blend。
- sourceから除外するもの:
  - `run_guarded_contact_override`の定義と呼び出し。
  - `ENABLE_GOLD_OVERLAY`以下のGold candidate/calibration/profile処理。
  - train-well ID一致による予測置換。
  - public output read/copy、固定sample SHA/row/well assert。
- 実装後にAST/text scanし、禁止symbolのcall countが0であることをmanifestへ保存する。

## 評価と判定

- 対応するhonest OOFがないため、CV improvementやpromotionを主張しない。
- exp497のstrict public-core OOF結果はrisk priorとして参照するが、weight fitには使わない。
- truth-freeで次だけを記録する。
  - SP45 vs Pipeline B、public pre-override vs exp413、final vs exp413のRMSE/MAE/p95/max差分。
  - well別、MD horizon別の差分量。
  - start continuity、prediction range/mean/std、fallback rows/wells。
- 差分readout、Public LB、source titleを見て0.10を変更しない。
- technical/reproducibility gate PASS時だけ第2枠candidateとする。外部提出は別承認。

## 再現性設計

- seed policy: `stable_sha256(split, family, well, seed_index)`。
- stochastic処理の有無: PF/likelihood-PFにあり。Beamは決定的処理として入力順を固定する。
- PF/Beam / likelihood-PF / seed bagging: public core内で使用。per-well seed bankを事前生成し、global RNGを使わない。
- 並列処理と乱数の関係: well単位seedをimmutable keyから生成し、thread/process schedulingと乱数列を分離する。
- CPU/GPU runtime: 保存model inferenceを基本にCPU/internet off。artifact欠落時の再学習はしない。GPU 0を設計値とし、実装時の実測runtimeを記録する。
- train cache / test feature regeneration SHA: raw testから生成するfeatureのrows/wells/schema/content SHAをfamily別に保存する。
- model manifest / prediction / submission SHA: mounted dataset version、全model SHA、source SHA、SP45/B/public/final prediction SHA、submission SHAを保存する。
- Kaggle package bootstrap: root/package/bootstrap内configのsource SHA、weights、Gold/contact禁止flag、dataset sourcesをreadbackする。
- deterministic anchor: stable seed実装後も初回runはfalse。同じkernel source/inputでのrerun prediction SHA一致後だけtrue候補。
- 公開artifact byte parity: stable seed化で変わり得るため目的外。構造・数式・cell境界parityを主張する。

## リスク

- リークリスク: source内にcontact/Gold処理が同居する。禁止symbol call scanとsidecar freeze順序で遮断する。
- CV/LB不一致リスク: Public LB 7.159はtitle/lineage情報で、独立したprivate-safe証拠ではない。第2枠hedgeに限定する。
- ランタイム/メモリリスク: raw hidden test上のPF/Beam再生成が重い。artifact fallback trainを禁止し、9時間上限に対する見積を実装前に作る。
- 再現性リスク: 原sourceはglobal RNG/seed揺れの履歴がある。stable per-well seed portを必須とし、byte parityを採用条件にしない。
- source解釈リスク: architecture説明と最終cellの入力が異なる。最終cellの`sp45_projection_submission.csv`を正として契約化する。

## Kaggle current-test実行設計

- 正規placeholderを上書きせず、実装済み候補notebookから同内容の
  `*_current_test_inference.ipynb`を生成する。
- 初回kernel id/title
  `kentookumura/exp510-exp413-exact-public-preoverride-hedge-inference` /
  `exp510 exp413 exact public preoverride hedge inference`は54文字slugとなり、Kaggle APIが
  詳細なし400で拒否した。50文字以下へ収めるため、同じexp510のcanonical execution slugを
  `kentookumura/exp510-exp413-exact-public-preoverride-inference` /
  `exp510 exp413 exact public preoverride inference`へ短縮固定する。
- runtimeはCPU、private、internet off。scientific variant 1、新規model/config/fold/booster、
  GPU、親再学習はいずれも0、保存Pipeline-B boosterは3。
- version 2はcompetition data、`fleongg/rogii-claude-models-pub` version 1、
  `kentookumura/exp413-scale5-likpf-current-test-inference`の公開test固定predictionをmountしたため
  hidden sampleとID不一致になった。修正版はこのkernel outputを外し、exp413 v4と同じ11 upstream
  kernel sourcesと保存75 modelからcurrent sample上でexp413を再生成する。
- 公開current test 3 wellsではPF 153 runs、beam 63 paths。hidden code rerunは動的well数で算出する。
- package生成後にmetadataとbootstrap manifestをreadbackし、config/source/model/input契約を確認する。
