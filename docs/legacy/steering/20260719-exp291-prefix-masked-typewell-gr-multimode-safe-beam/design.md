# 設計

## 仮説

same-well self-GRを除外し、Type Well GRのeligible local modesをsafe baseとともに保持して
複数checkpointの継続優位だけをcommitすれば、exp284のfalse switchを抑えてsafeを改善できる。

## アプローチ

既知 prefix の末尾640行を隠し、擬似 cut より前の128行だけで Type Well GR shift
likelihood を計算する。exp280 と同じ固定13 shift bank を使い、`abs(shift) >= 10 ft`
かつ隣接 slot 以上となる局所極大をすべて抽出する。exp284 のように1本へ早期確定せず、
各 mode を exp226 safe path の定数 shift branch として H128/H256/H512 まで全件保持する。

safe base は独立 branch として常に残す。代替 branch への commit は、同一 mode が H128 と
H256 の両方で safe より Type Well evidence が高く、両 checkpoint で固定 geometry veto を
通過した場合だけ許す。複数 mode が条件を満たした場合は現在checkpointのevidence最大を選び、
同点はshift bank順に固定する。H512 は同一条件を3 checkpointすべてで満たす永続性診断とする。
safe比較にはstrict `>` を用い、同点、非有限、veto失敗、候補なしはすべて safe に戻す。

post-cut truth は target-free candidate table、branch paths、checkpoint evidence、policy
selection の content SHA を凍結した後にだけ join し、RMSE、oracle headroom、false switch、
pairwise evidence を読む。

## 実験範囲

- 対象実験: `exp291_prefix_masked_typewell_gr_multimode_safe_beam`
- Route: `pf_beam`
- 親実験: `exp284_prefix_masked_wrong_mode_branch_recovery_backtest`
- 参照: exp226 geometry、exp209 Type Well GR emission、exp280 shift bank、
  exp281/exp283/exp285 negative evidence
- 変更する変数: candidate sourceを `self-GR top3 + Type Well top1` から
  `Type Well GR eligible local maxima 全件` へ変更し、safe-anchored multi-checkpoint commitにする。
- 固定する変数: mask 640、visible 512以上、visible score 128、H128/H256/H512、
  primary H256、1 cut/well、13 shift bank、minimum abs shift 10 ft、exp209 likelihood、
  exp284 geometry veto、exp226 fold-safe geometry replay、5-fold assignment。

## 固定候補契約

1. shift bank は `[-80, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 80] ft`。
2. pre-cut 128行の観測 `TVT_input` を参照座標として、exp209 Gaussian raw-GR
   Type Well mean log-likelihood を各 shift に計算する。
3. `abs(shift) >= 10 ft` の slot だけを eligible とする。
4. bank 全体で左右の score 以上なら局所極大。端点の欠けた側は `-inf` とする。
5. eligible local maximum をすべて保持し、shift値でdeduplicateする。top-K capは置かない。
6. local maximum がなければ safe-only。exp284 の highest-eligible fallback は使わない。
7. safe は shift 0 の独立 branch とし、候補順位に関係なく常に保持する。
8. same-well self-GR/NCC donor は一切生成しない。

## 固定 policy

- `safe_base_only`: safeを常に選択する control。
- `safe_plus_top1_typewell_mode`: visible likelihood 最大のlocal mode 1本だけを残す比較対象。
- `safe_plus_all_typewell_modes`: local modes 全件を残す主 policy。
- `safe_plus_matched_count_shuffled_modes`: eventごとの実候補数を保った deterministic shuffle control。

top1/all/shuffled のいずれでも safe保持、checkpoint、commit、veto、tie fallback は同一とする。
shuffleは各eventのreal local-mode数を `m` とし、scoreを見ずにeligible nonzero shiftから
`m` 本を重複なしでstable local RNG抽出する。real modeとの偶然の重複はそのまま許容する。

## 評価と停止条件

- technical guard: eligible 750 wells以上、5 folds、mask/safe/all-mode/finite coverage 1.0、
  pre-freeze truth access 0、self-GR candidate 0。
- evidence guard: 各fold AUC 0.60以上、pooled balanced choice accuracy 0.60以上、
  各fold accuracyは0.50超。
- primary guard: all-mode H256がsafeより0.10 ft以上、4/5 folds以上で改善。
- multi-mode value guard: all-mode H256がtop1より0.05 ft以上、3/5 folds以上で改善。
- safety guard: safe truth unique-best eventのfalse switch 5%以下、H512 gainはH256以上。
- negative-control guard: matched-count shuffleよりpooled改善、5/5 folds非悪化。

1つでも失敗したら branch を close する。同一 backtest の結果を見て K、候補cap、shift bank、
window、checkpoint、margin、likelihood sigma、veto を調整する rescue は行わない。

## 将来の成果物

- contract JSON
- mask manifest
- target-free shift likelihood / candidate bank
- target-free branch paths
- target-free checkpoint evidence
- target-free policy selection
- post-freeze overall/fold/pairwise/by-well metrics
- input/content SHA manifest
- summary JSON

prediction、model、submission は生成しない。

## 再現性設計

- seed policy: 実 candidate/policy はdeterministic。shuffle controlのみ
  `stable_sha256(well_id, cut, source, seed=42)` のlocal RNG。
- stochastic 処理の有無: matched-count shuffled negative controlのみ。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。routeは将来のbranch decoder先を表すが、
  本実験では有限候補のdeterministic score backtestだけを行う。
- 並列処理と乱数の関係: fold順、well_id順のsingle process。global RNGを使わない。
- CPU/GPU runtime と deterministic flags: CPU、worker 1、GPU/AMP off。
- train cache / test feature regeneration の SHA 記録方針: gzipはdecompressed content SHA、
  candidate/branch/evidence/selectionはschema SHAとcontent SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 非生成。contract/input/kernel versionと
  target-free artifact freeze SHAを記録する。
- Kaggle package bootstrap 確認方針: 実装時にmetadataとbootstrap内configを照合する。

## リスク

- リークリスク: post-cut truth を candidate selection に混入する危険が最大。mask後frameだけを渡し、
  target-free artifactsのfreeze後に別段階でtruthをjoinする。
- CV/LB 不一致リスク: known-prefix pseudo-tailはofficial suffixと条件が異なる。hidden-like spatial、
  hidden-like typewell-purged、prefix length bucketを診断し、passしても直接提出しない。
- false switchリスク: GR likelihoodがfold-stableでないことはexp284で確認済み。safeを絶対保持し、
  2-checkpoint継続優位をcommit条件とし、false switch guardをprimary safety条件にする。
- oracle候補リスク: modeを増やすとoracleは機械的に改善しやすい。safe/top1/shuffleとのpolicy成績を
  promotion条件にし、oracle headroomだけではpassさせない。
- ランタイム/メモリリスク: 候補数は固定13 bankで有界だが、branch tableをlong形式にすると増える。
  chunk保存し、全wellのbranch pathを同時保持しない。
- 再現性リスク: shuffleだけstable local RNG、処理順固定、content SHAで管理する。

## 実装メモ（2026-07-19）

- exp284 compact self-contained trainの10章構成とexp226 geometry replayを維持し、
  wrong-mode injectionとself-GR proposal部分だけをType Well local-mode bankへ置換した。
- real candidateはsafe 1本とeligible local maxima全件、negative controlは同じeventのreal countを
  保ったscore-blind fixed-bank sampleとし、両者を`control`列で分離した。
- H128は1 checkpoint、H256はH128/H256、H512はH128/H256/H512のすべてで同一branchが
  safeをstrictに上回り、各checkpointのvetoを通ることをcommit条件にした。
- pairwise evidenceはreal alternative対safeのH256 margin AUCと、H128/H256 persistent choiceの
  balanced accuracyとして実装した。
- canonical notebookは未採用であり、別名compact candidateだけを生成した。

## 次のアクション

canonical train採用とKaggle CPU version 1は承認後に完了した。technical guardは全PASSしたが、
H256 safe比-17.372335 ft、false switch 34.9462%、shuffle比悪化により総合FAILとなった。
同一truth上のparameter rescueを行わずbranchをcloseし、decoder、推論、submissionへ進めない。
