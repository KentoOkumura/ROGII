# exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit

## 状態

- ルート: `pf_beam`
- 状態: Kaggle private CPU version 2完了・scientific guard FAIL・branch close
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-20
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 比較契約: `exp293_physics_only_candidate_bank_headroom_contract`
- Kernel: `kentookumura/exp298-exp226-local-shape-quotient-audit-train` version 2 / id_no `127956072`
- 利用可否: local-shape source不採用。後続Stage 2/3/4へ進めない

## 仮説

exp226は大局的offset/driftのためoverall RMSEでは不利でも、`tvt_geop + gr_delta`の局所形状は
exp293 deployable12より強い可能性がある。H256/H512 blockごとにtrue residualの定数項・一次傾向を
診断上だけ除去し、残る形状誤差を比較する。

## 変更点

- exp226 OOFを7列allowlistで読むloaderを実装し、`tvt_true/error/abs_error`をpre-freezeでmaterializeしない。
- `P_geop=tvt_geop`、`P_preU=tvt_geop+gr_delta`、`P_postU=tvt_pred`をfloat64で再構築する。
- exp293の固定12候補と固定block assignmentを比較基準にする。
- candidate/component/blockをtruthなしでSHA freezeし、その後だけtrue suffix TVTを接続する。
- offset/affine quotient、一次差/二次差、fold/scope/block/by-well readoutと固定PASS判定を実装した。
- oracle offset/slopeは集約式の中だけで除去し、係数や補正predictionを生成・保存しない。
- compact self-contained train候補とfail-closed inference候補、専用testを追加した。
- Lateフェーズ固有の設計は含めない。
- 学習、PF/Beam再生成、推論、提出は行わず、Kaggleでは固定auditだけを実行した。

## 検証方針

- Fold: exp263/exp293の保存済み5 foldsを再利用、再fitなし。exp226 source foldは別割当なのでcrosswalkを
  provenanceとしてfreezeするが、評価foldには使わない
- Group: well
- Primary: H256/H512 pooled affine-quotient RMSE、候補rank、fold rank、unique-best block比率
- Secondary: H128/whole-well、offset-only、一次差/二次差、1000+、hidden-like 2面、by-well p95/worst
- Leakage Check: pre-truth freeze SHA、truth別loader、oracle係数/補正path非保存、alias重複排除
- PASS: `config.yaml`とsteeringの全technical/scientific guardを満たした場合だけ

## 設計の正

- exp298本体: `docs/legacy/steering/20260720-exp298-exp226-blockwise-offset-slope-quotient-local-shape-audit/`
- 後続案2・3・4: `downstream_branch_contract.md`

後続契約は、2のfixed `S512` hybrid bank、3のlatent-registration semi-Markov、4のnested block rankerを
独立実験として固定している。変更にはユーザーの明示承認が必要である。

## 実行入口

compact train候補はユーザー承認後に正規train Notebookへ採用した。正規inference Notebookは変更せず、
fail-closedのまま維持している。実装のJupytext sourceと候補は次の別名ファイルである。

- `exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_compact_selfcontained_train.py/.ipynb`
- `exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_compact_selfcontained_inference.py/.ipynb`

Kaggle private CPU version 2は完了済みである。inferenceはfail-closedのまま採用・実行しない。

## 生成物

Kaggle実行でinput/component/freeze manifest、pooled/fold/scope/block/by-well metrics、summary、SHA manifestを
生成した。小規模metrics/manifestを選択取得しSHAを照合済み。oracle係数、offset/slope補正prediction、model、
submissionは生成していない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| implementation status | complete / Kaggle audit complete |
| H256 affine RMSE / rank | 0.3482675905 / 4 |
| H512 affine RMSE / rank | 0.7224085771 / 5 |
| H256 / H512 post-U RMSE | 0.3041197991 / 0.6096467779 |
| top3 folds H256 / H512 | 0/5 / 0/5 |
| technical / scientific | PASS / FAIL |
| dedicated tests | 11 passed |

## 所見

### 良かった点

- exp226をoverall RMSEだけで棄却せず、局所形状sourceとして直接検証できる問いへ分離した。
- oracle quotientをdeployable correctionと混同しないfreeze/非保存契約を固定した。
- singleton除外を含むtechnical guardは全PASSし、scientific FAILを実装問題から分離できた。

### 悪かった点

- `P_preU`はH256/H512で4/5位、全foldでも4/5位、1000+とhidden-like 2面も5位だった。
- post-UよりH256 `+0.044148 ft`、H512 `+0.112762 ft`悪く、局所source仮説は支持されなかった。

### リスク / 注意

- `P_postU`はexp293の`exp226_k16`と同一aliasなのでrankへ二重投入しない。
- H128だけの改善、単一fold、単一scopeをPASS根拠にしない。
- exp226 OOFのsuffix長preflightでは、最終block長1がH128/H256/H512で`4/2/2 wells`存在する。承認済み契約では
  exp293境界を維持し、singletonをaffine metric/rank/win/unique-best分母から全候補共通で除外する。
- technical coverage 1.0は長さ2以上のaffine-eligible rowsに要求し、singleton数を生成物へ必ず記録する。
- exp293/exp297のfixed12契約とexp295独立SSMを変更・統合しない。

## 次

固定契約どおり本枝を閉じる。Stage 2/3/4、救済grid、inference、submissionへ進まない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
