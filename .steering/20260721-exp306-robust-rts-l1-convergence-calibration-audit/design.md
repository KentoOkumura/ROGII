# 設計

## アプローチ

exp304のscientific FAILを救済せず、solverが固定objectiveに対して技術的に収束可能かだけを2段階で監査する。Stage 0はtarget-freeな固定64 wellsで反復予算とRTSの事前分岐許容差を判定し、Stage 0を全件通過したbranchだけを全773 wellsへ展開する。denoised signalの科学的品質は本実験では読まない。

## 仮説

RTS/L1のexp304 failureは反復上限内の未収束であり、objectiveを変えずに反復予算と事前登録したRTS許容差だけを調整すれば全件収束可能である。

## 実験範囲

- 対象実験: `exp306_robust_rts_l1_convergence_calibration_audit`
- Route: `pf_beam`
- 親実験: `exp304_gr_denoiser_emission_separability_readout`
- 変更する変数: RTSの最大IRLS反復数と条件付き許容差、L1 ADMMの最大反復数だけ。
- 固定する変数: raw input/common missing policy、series ordering、coordinate normalization、solver objective、RTS Q/R/df/scale floor、L1 lambda/rho/tolerance、technical gate。
- 除外: truth join、shift score、MRR/top3、RMSE、HMM/PF/Beam、inference、submission、exp304 selection更新。

## exp304 failure anchor

- robust RTS: `max_irls=8`, relative mean tolerance `1e-6`で1,531/1,546 series FAIL。
- L1 trend: `rho=1`, `max_admm=500`, abs/rel tolerance `1e-4`で974/1,546 series FAIL。
- exception、NaN、silent fallbackが原因ではなく、反復上限内の未収束でtechnical FAILした。
- raw well-file identity content SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`。
- exp304 scientific contract content SHA: `8822df968200b74ea9969b0bc023ec127debbff01933bdc89ff3db9844d55064`。
- exp304 solver statusとmanifestを診断参照にできるが、truth/readout列は入力しない。

## 固定solver契約

### robust RTS

- level+slope state、median positive spacingによるcoordinate normalization。
- Student-t IRLS df 4、first-difference robust measurement scale、second divided-difference acceleration scale。
- constant-velocity transition、discrete white acceleration process covariance、exp304と同じinitial mean/covariance、finite scale floor `1e-6`。
- Candidate A: `maximum_iterations=32`, `relative_mean_tolerance=1e-6`。
- Candidate B: `maximum_iterations=32`, `relative_mean_tolerance=1e-4`。AがStage 0で128/128を満たさない場合だけ実行する。
- AがStage 0をPASSした場合はBを実行しない。A/Bをscience scoreで比較しない。

### L1 trend

- objective: `0.5 * ||y-x||_2^2 + lambda * ||D2 x||_1`。
- lambda: exp304と同じfirst-difference MADとseries lengthから決まる式。
- ADMM `rho=1`, abs/rel tolerance `1e-4`を固定し、`maximum_iterations=2000`だけを変更する。
- adaptive rho、warm-start別解、lambda/tolerance変更は行わない。

## 検証方法

### Stage 0: target-free calibration

1. raw trainの773 well IDsを列挙し、各wellについてlowercase hex SHA256 of UTF-8 `exp306-stage0-v1|<well_id>`を計算する。
2. `(sha256, well_id)`昇順の先頭64 wellsを固定sampleとし、sample manifest/content SHAを保存する。
3. 水平well CSVは`TVT`を読み込まず、`MD/GR/TVT_input`だけを使用する。typewellは`TVT/GR`をsolver座標・信号として使う。
4. exp304と同じtarget-free preparationでhorizontal/typewellの計128 seriesを作り、input content SHAを凍結する。
5. RTS AとL1を実行する。RTS Aが1 seriesでもFAILした場合だけRTS Bを実行する。
6. 128/128 convergence、finite、length/order、fallback、iteration/status、runtimeをbranchごとに判定する。
7. Stage 0 sampleの先頭8 wellsを同一process/thread設定で再実行し、output/status content SHA完全一致を確認する。
8. branch別full runtimeを`stage0_elapsed_seconds / 64 * 773`で外挿し、8.5時間以内だけをfull eligibleとする。

Stage 0 core実行量はRTS A + L1の256 series-runs、RTS B条件成立時は最大384 series-runs。deterministic parity rerunはfull eligible branchごとに16 series-runs、最大32とする。

### Stage 1: full technical audit

- RTSとL1のeligible branchを別々のKaggle CPU run/versionで監査し、各runのruntime上限を8.5時間とする。
- 各branchは773 wells x horizontal/typewell = 1,546 series-runs。2 branchともeligibleなら最大3,092 full series-runs。
- technical PASSは1,546/1,546のconverged、finite input/output、length/order一致、silent fallback 0、input/output/status manifest/SHA一致のANDとする。
- Stage 0でRTS AがFAILしBがPASSした場合、fullはBだけを使う。Aをfull実行しない。
- fullで失敗したbranchだけを閉じ、他branchの独立判定を維持する。
- full outputは将来の科学評価へのeligibility evidenceに限り、exp304 metricsやselected SWTを変更しない。

## 生成物契約

- scientific contract JSON
- Stage 0 sample manifest CSV/JSON
- Stage 0 input/output/solver status CSV.gz
- Stage 0 branch gate/runtime projection JSON
- deterministic parity manifest JSON
- branch別full input/output/solver status CSV.gz
- branch別full technical gate/runtime JSON
- summary JSON

truth、error、formation、MRR/top3、RMSE、prediction、submissionは生成物へ含めない。output CSV.gzはraw gzip SHAとdecompressed content SHAを分けて記録する。

## 再現性設計

- seed policy: RNGなし。sampleは固定salt付きSHA256順で決める。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理と乱数の関係: RNGなし。Stage 0とparity rerunは同一の固定thread設定で実行し、well結果はwell_id/series_kind順に書く。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/internet off。Stage 0は1 run、fullはeligible branchごとに別run、各full run 8.5時間上限。
- input/output SHA: raw well-file identity、prepared series、denoised output、solver status、sample manifestのcontent SHAを保存する。gzipはdecompressed SHAを主証拠にする。
- model/prediction/submission SHA: すべて非該当。本実験はsolver technical auditのみ。
- Kaggle bootstrap: 実装・push承認後に正のconfig/sourceから再生成し、bootstrap内stage、solver settings、approval flagを照合する。
- deterministic anchor: submission anchorではない。固定input/runtimeに対するtechnical reproducibilityだけを主張する。

## リスク

- リークリスク: solver調整にtruth/error/scientific scoreを使うとposthoc rescueになる。水平wellの`TVT`をload時点で禁止し、score joinを実装しない。
- 科学的リスク: 収束しても有用なdenoiserとは限らない。PASSは別expの科学評価資格だけでありexp304選択を変更しない。
- 分岐リスク: RTS A/Bが実質gridになる可能性がある。A FAIL時のみBを実行し、B FAIL後の追加tol/iteration調整を禁止する。
- ランタイムリスク: 反復上限増加でfull runが長くなる。Stage 0外挿とbranch別8.5時間hard gateでfail-closeする。
- 数値再現性リスク: BLAS/library/thread差で停止iterationが変わり得る。version/threadを固定し、8-well exact parityを必須にする。
- 運用リスク: exp305より先に実行すると高優先decoder仮説を遅らせる。本実験は設計済みに留め、実行順序はexp305とexp276を優先する。

## 次のアクション

Stage 0実装は完了した。Kaggle Stage 0は別承認まで停止し、full auditはStage 0 evidenceとさらに別の実装・実行承認を確認してから行う。
