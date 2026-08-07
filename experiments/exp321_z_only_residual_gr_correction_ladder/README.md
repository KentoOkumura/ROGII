# exp321_z_only_residual_gr_correction_ladder

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU Run AB v1完了、Stage A PASS / Stage B FAIL、Stage C閉鎖
- CV / Public LB / Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-21
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

最後の既知`TVT_input`をanchorとして`TVT`を固定`-ΔZ`で延長すれば、未知suffixの局所形状の大半を説明できる。残る低周波offsetにType Well GRとの対応があれば、exp226 window GR correctionをZ-only経路へ加えることで安全に改善できる。

```text
tvt_z(t) = anchor_tvt - (Z(t) - anchor_z)
tvt_z_gr(t) = tvt_z(t) + gr_delta_exp226_window(t)
```

## 3段階gate

1. Stage A: Z-only残差がH256/H512で低次元offset/slopeとして説明でき、`±4 ft`補正にheadroomがあるか読む。
2. Stage B: exp280と同じ固定13 shift・512行block・GR likelihoodで、Z-only周囲のtruth-nearest shiftをshuffleおよびexp280より良く順位付けできるか読む。
3. Stage C: A/B両PASS後の別runで、exp226 window GR correctionのbaseだけをZ-onlyへ置換し、Z-only比RMSEとtail safetyを評価する。

Stage A/Bはtarget-free path/scoreを先に凍結してから採点する。Stage Cも別runでpredictionを凍結してから採点する。

## 検証方針

- Fold: exp226保存5 foldsを再利用し、学習や再分割は行わない。
- Group: `well_id`。期待値は773 wells / 3,783,989 suffix rows。
- Stage A/B: target-free tableとscoreをSHA freezeした後だけtrue TVTを結合する。
- Stage C: A/B全PASS後の別runでpredictionをSHA freezeした後だけtrue TVTを結合する。
- 評価面: overall、5 folds、near 0--250 ft、1000+、hidden-like spatial、hidden-like typewell-purged、by-well p95/worst。
- Leakage check: suffix truth/error/oracleをprediction、shift score、block、feature、gate入力に使わない。

## 固定した範囲

- Z係数`-1`、last-known anchor、exp226保存5 folds。
- donor、XY、ANCC、kappa、U projection、learned slope/intercept/rateは不使用。
- Stage Bはexp280 parity、Stage Cはexp226 GR correction parity。
- model、booster、HMM、PF、Beam、inference、submissionはなし。
- 最大実行量はStage ABが1 diagnostic / 5 fold strata / 0 booster、Stage Cが1 candidate / 773 window-decoder well-runs / 0 booster。

## 実行入口

- trainの正規Notebookとcompact版は、`exp321_z_only_residual_gr_correction_ladder_compact_selfcontained_train.py`をJupytext sourceとして生成済み。
- Stage A/Bは1 diagnostic contract、5 fold strata、0 model / 0 trained fold / 0 booster / 0 HMM / 0 window decoderで実行した。
- `execution_contract.kaggle_push_approved=true`。先行exp305完了後、canonical kernel version 1をKaggle CPU / internet offで完了した。
- inference Notebookもfail-closedで、predictionやsubmissionを生成しない。
- Stage Cは未実装。Run ABでStage Bが固定gateをFAILしたため、実装・実行せず閉じた。

## 実装済みの生成物契約

- target-free: Z-only path + H128/H256/H512 block identity、13 shift GR likelihood score、stable shuffled score、schema/content SHA。
- late truth Stage A: direct / offset quotient / affine quotient、lag-1、block mean/slope、cap4 oracle diagnostic、固定gate。
- late truth Stage B: top1/top3/MRR/sign、shuffle/exp280比較、fold/1000+/hidden-like、bank/quantization coverage、固定gate。
- decision: A/B両PASS時だけStage Cの別承認を許すdecision manifest。FAIL後のthreshold/shift/sigma救済はしない。

## 条件付き後続

案4 `z_only_residual_offset_exact_hmm_probe`と案5 `z_only_gr_sparse_candidate_addonly`は、[reserved_followups.md](reserved_followups.md)にtriggerと禁止事項を固定した。Stage B FAILとStage C未到達によりtrigger不成立となり、どちらも未採番・未実装のまま閉じた。

## Run AB結果

- Kaggle kernel: `kentookumura/exp321-z-only-residual-gr-ladder-train` version 1
- runtime: 611.963秒、3,783,989 rows / 773 wells / 5 folds、1 diagnostic、model / booster / HMM / window decoder `0 / 0 / 0 / 0`
- Stage A: PASS。H512 affine quotientはZ-only `0.609237`、exp226 `tvt_geop` `0.669091`、比`0.910543`。5/5 foldsがrelative-shape gateを満たし、affine SSE説明率`0.999968`、cap4 oracle gain`3.205124 ft`。
- Stage B: FAIL。top1/top3/MRR/signは`0.332991 / 0.587903 / 0.503399 / 0.685887`で、shuffleを5/5 foldsと全stress scopeで上回った。一方、固定`[-80,80] ft` bankのrange coverageは`0.494029`、quantization coverageは`0.604212`、最大量子化誤差は`384.734576 ft`で固定gateをFAILした。
- decision: `close_stage_c_branch_without_rescue`。bank、sigma、threshold、decoderの事後救済は行わない。

## 所見

- Z-only残差はH512内ではほぼaffineだが、direct RMSEは`107.494824 ft`、block mean絶対値は`90.628894 ft`で、固定小補正の前提よりoffset scaleが大きい。
- GR likelihoodの順位信号自体は強く再現したが、固定bank外のblockが約半数あるため、exp226 window GR correctionへ進む十分条件を満たさない。
- exp280/281/298と合わせ、同一truth上でbankや補正幅を広げる救済は優先しない。

## 次

Stage C、inference、submission、予約案4/5を閉じる。同系のparameter rescueを追加せず、独立した既存優先実験へ戻る。
