# 設計

## アプローチ

### 共通のZ-only経路

raw horizontal fileの行順を維持し、最後の有限`TVT_input`行をanchor `s`とする。suffixの各行で次を決定的に計算する。

```text
anchor_tvt = TVT_input[s]
anchor_z   = Z[s]
tvt_z[t]   = anchor_tvt - (Z[t] - anchor_z)
relpath_z[t] = tvt_z[t] - anchor_tvt = -(Z[t] - anchor_z)
```

これは`U=TVT+Z`をanchor以降一定とする経路であり、係数、offset、rateをfitしない。`TVT_input`がprefix内で連続せずsuffix後に再出現するwell、anchorの`Z`が欠損するwell、suffix `Z`が欠損するrowはsilent fallbackせずtechnical FAILとする。

### Stage A: Z-only residual structure readout

`tvt_z`、row identity、exp226 fold、suffix offset、H128/H256/H512の非重複block割当をtruth-freeに作り、content SHAを凍結した後だけtrue TVTを結合する。残差`r=tvt_true-tvt_z`について、direct RMSE、block meanを除いたoffset quotient、block内row indexへのOLS切片・傾きを除いたaffine quotient、lag-1 correlation、block residual mean/slopeを読む。末尾short blockは保持し、affine metricでは2行未満を全候補共通で除外する。

Stage Aは次をすべて満たす場合だけPASSとする。

- H256/H512のaffine-quotient RMSEが、同じblock上で再計算した保存exp226 `tvt_geop`の1.25倍以下。
- 上記relative shape guardを各horizonで4/5 folds以上満たす。
- H512でaffine quotientがZ-only direct residual SSEの80%以上を説明する。
- H512のblock mean residualを`[-4,+4] ft`へclipしたoracle diagnosticがZ-only RMSEを0.05 ft以上改善する。
- row/well/fold/finite/anchor coverageがすべて1.0。

oracle offset/slope/cap4値は診断専用であり、prediction、feature、gate入力として保存しない。FAIL時もStage Bのreadout生成物は診断記録として保存できるが、Stage Cは開始しない。

### Stage B: Z-only GR shift likelihood separability readout

exp280のbase pathだけを`tvt_z`へ置き換える。

```text
candidate_tvt(t, delta) = tvt_z(t) + delta
delta in [-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80]
```

未知suffixを先頭から非重複512行blockに分け、末尾short blockを保持する。exp209 Gaussian raw-GR/typewell emission、known-prefix residual std clip `[10,60]`、typewell extension 40 ft、missing-GR処理、mean row log likelihood、config順tie breakをexp280から固定継承する。real scoreはRNGなし、negative controlだけをwell/blockごとのstable SHA256 local RNGで13 candidate score permutationする。

truth-nearest shift、top1/top3/MRR/sign、margin、regret、bank range/quantization coverageはtarget-free score SHA凍結後にだけ計算する。Stage Bは次をすべて満たす場合だけPASSとする。

- top1/top3/MRR/signのrealがstable shuffleを各5/5 foldsで上回る。
- pooled top1/top3/MRR/signがexp280保存値`0.189547 / 0.452421 / 0.389626 / 0.498523`を4指標すべて上回る。
- 1000+、hidden-like spatial、hidden-like typewell-purgedで4指標のreal-shuffle差がすべて正。
- bank range coverage 0.95以上、最大量子化誤差20 ft以下、finite/row coverage 1.0。

同値はPASSにしない。FAIL後のshift、block、sigma、likelihood、threshold救済は行わない。

### Stage C: exp226 window GR correction on Z-only

Stage A/BがともにPASSした場合だけ、別のKaggle CPU train runで1候補を生成する。exp226 `gr_correction`の数式・定数を固定し、次の2入力だけを置換する。

```text
geop   := tvt_z
relpath := relpath_z
tvt_z_gr := tvt_z + gr_delta_exp226_window
```

固定値は`grid_step=0.5`、`window=500`、`stride=125`、`tau=2.0`、`w_mse=0.5`、`w_level=0.1`、`shrink_a=1.1`、`shrink_b=0.12`、`shrink_floor=0.3`、`cap=4.0 ft`、`s0=0.10`、`extent=30 ft`である。known-prefix affine calibrationとsigma clip `[5,60]`、prefix pair 30以下でsigma 30、GRの最大10行interpolationもexp226と同じにする。`enable_u_projection=false`、donor/XY/ANCC/kappaは不使用とする。

exp226の広い`except -> zero`は科学式ではないため移植せず、unexpected exceptionはrun全体をtechnical FAILにする。明示的なineligible条件（typewell不足、window centerなし、finite profile不足）はzero correctionへsafe fallbackし、reason別well/row coverageを必ず記録する。

Stage C prediction、correction、status manifestをtruth-freeに凍結してSHAを取得した後だけsuffix truthと保存controlを結合する。科学的PASSは`tvt_z`比overall gain 0.05 ft以上、4/5 folds、near 0--250 / 1000+ / hidden-like 2面のRMSE非悪化、by-well p95/worst delta 0.25 ft以下、finite/identity/cap coverage 1.0をすべて必須とする。将来のroute候補扱いには、さらに保存exp226 final OOF RMSE 9.427110以下を要求する。exp263 fixed 8.238332は参考比較として報告するが、exp321の科学的gateには混ぜない。

## 実行単位と真値境界

- Run AB: Stage Aの`tvt_z`/blockとStage B shift scoreを、truth列をロードせず生成・SHA freezeする。その後だけtruthを1回結合し、Stage A→Stage Bの順でgateを判定する。最大1 diagnostic contract、5 fold strata、model/booster/HMM/window decoder 0。
- Run C: Run ABのdecision manifestがA/BともPASSした場合だけ別runで実行する。truth-freeに`tvt_z_gr`を生成・SHA freezeし、その後だけ採点する。1 candidate、最大773 window-decoder well-runs、model/booster/HMM/PF/Beam 0。
- Run AB/CともCPU、internet/GPU/TPU off。inference/submissionはdisabled。

## 実験範囲

- 対象: `exp321_z_only_residual_gr_correction_ladder`
- Route: `pf_beam`
- 科学的親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 参照: exp206、exp213、exp264 physical summary、exp280、exp281、exp298。
- 変更する変数: exp226/exp280のbase pathを、最後の既知TVTから固定`-ΔZ`で延長した`tvt_z`へ置換する。
- 固定する変数: raw rows、exp226 fold、anchor規則、shift bank、GR emission、window correction全定数、scope、gate、tie/fallback、truth join policy。
- 除外: learned `a,b`、prefix slope/rate fit、XY/donor/ANCC/kappa、U projection、exact HMM、PF/Beam、ML/selector、blend、inference、submission。

## 再現性設計

- real path/correctionはRNGなし。stable shuffleだけを`SHA256(experiment, seed, well, block)`から作るlocal RNGで固定する。
- raw file、typewell content、exp226 OOF、hidden-like assignment、fold/row/block manifest、target-free shift score、predictionのschema/content SHAを記録する。
- gzipはraw SHAとdecompressed content SHAを分け、後者を主証拠とする。
- well文字列、raw row、block、shift、window centerは固定順。global RNGとPython `hash()`は禁止する。
- model/model manifest/submission SHAは非該当。Run Cではprediction raw/decompressed/logical content SHAを記録する。
- Kaggle package作成時はcanonical kernel id、CPU/internet-off、source/loose/bootstrap内configのbyte parityを確認する。
- fixed-input diagnostic/candidateであり、inference/submission rerunまで行わないためdeterministic submission anchorとは呼ばない。

## リスクと停止条件

- exp206のlearned slopeは長期累積driftで失敗したため、本実験では係数`-1`から動かさない。
- exp213のprefix structural priorはPFで悪化したため、Z-onlyをtransition priorやPFへ入れない。
- exp280はlikelihood信号を示したがtop1約19%に留まり、exp281はtailを大幅悪化させたため、hard shiftと常時稼働exact HMMをexp321へ含めない。
- exp298ではexp226局所形状仮説が固定bank比でFAILしたため、Stage AでZ-only局所形状を独立に反証してから補正へ進む。
- Stage AまたはBがFAILならRun Cを実装済みでも実行せず、同一truthでthreshold/gridを救済しない。
- Stage Cが科学的PASSでもexp226 9.427110を上回れなければ、inference、candidate追加、submissionへ進まない。

## 条件付き後続案4/5の固定境界

### 案4: `z_only_residual_offset_exact_hmm_probe`（未採番・高リスク）

exp321 Stage BとStage C科学的gateの両方がPASSした場合だけ別expとして設計可能とする。状態は`tvt=tvt_z+delta`、decoderはexp281の`delta [-80,80] / step 0.35 / 41 rate states / rate span ±0.10 / sig_r 0.002 / sig_p 0.02 / momentum 0.998`を1 variantだけ固定継承する。baseをexp226 `tvt_geop`から`tvt_z`へ変える以外のgrid/process-noise/likelihood救済は禁止する。exp321、exp281、exp226、exp263 saved controlと比較し、常時稼働HMMのtail再発を主リスクとする。exp321には実装しない。

### 案5: `z_only_gr_sparse_candidate_addonly`（未採番・条件付き）

exp321 Stage Cが科学的PASSかつexp226 9.427110以下を満たした場合だけ別expとして開始可能とする。selector候補へ追加するのは`tvt_z_gr` 1本だけで、`tvt_z`は診断controlに限定する。まずexp293/修正版exp264 fixed12に対するH512/whole-well add-one oracle、strict unique-best、残差相関を0 modelで監査し、oracle gainが両面0.05 ft以上・5/5 folds・strict unique-best 5%以上の場合だけ、exp286と同じ13番目add-only contractでsaved exp264 controlを再学習せずselectorを学習する。selector/downstream model数とGPU/CPUコストは実行前に別途承認を得る。hard replacement、2候補同時追加、threshold router、inference、submissionは先行gate前に行わない。exp321には実装しない。

