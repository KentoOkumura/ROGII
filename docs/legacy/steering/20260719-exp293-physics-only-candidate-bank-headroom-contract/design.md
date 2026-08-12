# 設計

## 目的と判断対象

物理モデル単体の現行anchorはexp263の固定formulaで、OOF RMSE `8.2383315465`、Public LB `7.800`。
exp286のOOF 13候補unionではrow / H512 / whole-well oracleが約`3.32 / 3.56 / 4.69`だったが、
`geop_hmm`はcurrent-test生成契約が未完成である。exp293ではLB 6.5への到達可能性を過大評価しないため、
exp263 inferenceで実際に再生成済みの12候補だけをprimary bankとしてoracle headroomを再確定する。

exp293は「どのcandidateを選ぶか」を検証しない。現在のdeployable bank内に6.5を支える候補があるか、
候補選択をrowより物理的なblock単位へ制約してもheadroomが残るかだけを判定する。

## Primary candidate bank

候補順は次で固定する。

1. `exp226_k16`
2. `selfgr_hmm_a070`
3. `likpf_mean`
4. `exact_hmm`
5. `pf_ancc`
6. `beam_mean`
7. `exp226_k16__selfgr_hmm_a070 = 0.5*exp226_k16 + 0.5*selfgr_hmm_a070`
8. `exp226_k16__exact_hmm = 0.5*exp226_k16 + 0.5*exact_hmm`
9. `exp226_k16__likpf_mean = 0.5*exp226_k16 + 0.5*likpf_mean`
10. `selfgr_hmm_a070__likpf_mean = 0.5*selfgr_hmm_a070 + 0.5*likpf_mean`
11. `likpf_mean__exact_hmm = 0.5*likpf_mean + 0.5*exact_hmm`
12. `exp226_w500_50_50 = 0.5*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm`

6 primitiveはexp263の保存OOFを読む。5 pairとfixed formulaは保存済みexp263 contractと同じfloat演算順で
再構成し、代表sampleと全体RMSEのparityを確認する。train-only candidate、ML output、oracle由来candidate、
exp286 `geop_hmm`はprimary bankへ加えない。

## Oracle定義

- row oracle: 各rowで二乗誤差が最小の候補を選ぶ。tieはcandidate順だがSSEには影響しない。
- block oracle: 各wellのprediction suffix先頭から非重複H128/H256/H512 blockを作り、block SSEが最小の
  1候補をblock全行へ適用する。最後のshort blockも保持する。
- whole-well oracle: wellのprediction suffix全体のSSEが最小の1候補をwell全行へ適用する。
- oracle prediction自体は保存しない。candidate choice、SSE集計、by-well/by-block readoutだけを保存する。
- pooled RMSE、5 folds、MD distance bucket、1000+、hidden-like spatial/typewell-purgedを出す。
- anchor/target/oracleのSSEから次を計算する。

\[
R_{6.5}=\frac{SSE_{anchor}-SSE_{6.5}}{SSE_{anchor}-SSE_{oracle}}
\]

これは後続モデルがLB 6.5相当へ到達するために回収すべきoracle SSE headroomの比率として記録する。

## Support判定

technical PASSは次をすべて満たすこと。

- 3,783,989 prediction rows、773 wells、5 folds。
- 12候補すべてでrow identity一致、duplicate 0、finite coverage 1.0。
- exp263 formula parity最大絶対差`<=0.001 ft`。
- candidate bank manifestとoracle join前candidate content SHAが保存されている。
- candidate freeze前にtrue TVT、target、error、abs_error、oracleを読んだ回数が0。

scientific support PASSは次をすべて満たすこと。

- primary H512 pooled oracle RMSE `<=5.5 ft`。
- H512 oracle RMSEが5 foldsすべてで`<6.5 ft`。
- H512 oracleがanchorより5 foldsすべてで改善する。
- `R_6.5`がfiniteかつ`<=1.0`。

1000+とhidden-like 2面はrisk flagとして必ず記録するが、primary分岐を後から都合よく変更しないため、
support PASS/FAIL自体は上の4条件だけで決める。risk flagは第2/3段階の必須subgroup guardへ継承する。

## 固定分岐

- support PASS: 第2段階`prefix_calibrated_latent_registration_gr_evidence`へ進む。第4段階は開始しない。
- support FAIL: 第4段階`mode_loss_triggered_candidate_birth_beam`へ進む。第2/3段階はbank再監査まで開始しない。
- 第2段階PASS: 第3段階`joint_physics_candidate_registration_semimarkov_smoother`へ進む。
- 第2段階FAIL: 第3段階へ進まず、第4段階にも自動分岐しない。候補不足ではなく観測識別性不足として閉じる。
- 第4段階でbankを拡張した場合: exp293と同じoracle contractを別実験で再実行し、support PASS後に第2段階へ戻る。

後続の詳細な固定契約は実験側`downstream_branch_contract.md`を正とする。

## 実験範囲

- 対象実験: `exp293_physics_only_candidate_bank_headroom_contract`
- Route: `pf_beam`
- 親実験: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 主要参照: exp286 oracle audit、exp281/290 offset failure、exp291 multimode、exp292 warp-rate。
- 変更する変数: 評価粒度だけ。row / H128 / H256 / H512 / whole-well oracleを同一bankで比較する。
- 固定する変数: 12候補、candidate値/formula、fold、evaluation suffix、anchor、目標6.5、subgroup定義。
- 実装/実行範囲: compact train/inference、contract tests、canonical train採用、Kaggle CPU auditまで完了。

## 再現性設計

- seed policy: primary oracleはRNGなし。well/fold/row/candidateをstable sortする。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed baggingの有無: 保存済み候補を読むだけで再生成0。
- 並列処理と乱数の関係: 初回実装はsingle process。将来並列化してもreduction順を固定する。
- CPU/GPU runtime: Kaggle private CPU、GPU/AMP/internet off。version 2を約200秒で完了。
- input SHA: exp263 cache manifest/catalog、各primitive gzipのraw/decompressed SHA、fold map SHAを記録する。
- feature/content SHA: primary12 candidate-long logical content、bank manifest、block assignment、oracle readoutを記録する。
- model/prediction/submission SHA: model、selected prediction、submissionを生成しないため対象外と明記する。
- Kaggle bootstrap: package作成時にloose/package configとnotebook bootstrap内config SHAを比較する。
- deterministic anchor: false。固定入力へのdeterministic diagnosticでありsubmission anchorではない。

## リスク

- リークリスク: oracle計算にはtrue TVTが必要。candidate bankとblock assignmentをfreezeしてから別joinする。
- deployabilityリスク: exp263 OOF core12とStage 1 deployable12を混同しない。本実験は6 primitive+5 pair+1 fixedだけ。
- CV/LB不一致リスク: exp263のOOF 8.238に対してLB 7.800だがtestは3 wells。6.5到達判定をLB差だけで緩めない。
- oracle過大評価: row oracleだけではなくH512とwhole-wellを必須にし、primaryはH512とする。
- formula重複: `likpf_mean__exact_hmm`と別名w500 aliasを二重候補化しない。12番目は3成分fixed formulaだけ。
- ランタイム/メモリ: 3.78M×12をwideで常駐させず、candidate/block SSEをchunk集約する設計とする。
- 再現性: float32/float64演算順でpair parityがずれる可能性があるためexp263 contract順と0.001 ft toleranceを固定する。

## 実装確定

- trainはJupytext percent形式のcompact self-contained候補とし、同一exp helperをimportしない。
- candidate valuesはfloat32 memmap、oracle SSEはcandidate×row chunkで集約し、12列の二乗誤差wide matrixを作らない。
- block assignmentとcandidate contentをfreezeした後、bankを再hashしてからraw train truth loaderを開く。
- block assignment gzipはraw/decompressed/logical SHAを記録する。
- inference候補はfail-closedとし、raw test、prediction、submissionを作らない。
- compact trainを正規notebookへ採用済み。inferenceは未採用・disabledのまま保持する。

## 実行結果による分岐確定

- H512 pooled oracle RMSE `3.683763`、全fold `<6.5`、全fold anchor改善、必要回収率`0.471825`。
- technical/scientific supportはPASS。
- candidate不足を理由とするStage 4は開始しない。
- 次はStage 2 `prefix_calibrated_latent_registration_gr_evidence`だけを別実験として設計する。
