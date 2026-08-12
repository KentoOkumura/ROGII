# 要件

## 依頼

exp302で得るK12/K16/K24予測のtarget-free安定性が、corrected exp264 Stage C v6でK16を誤rankingする
領域を識別できるか監査する。バックログ、実験ディレクトリ、steeringを作成して設計を確定する。
初回設計時は実装を保留した。2026-07-21に4 dependency成立とユーザーの実装依頼を確認し、
別名compact self-contained source/Notebookの実装と静的検証までをscopeへ追加した。
同日、ユーザーの実行依頼により正規Notebook採用と1回のprivate CPU package/push/run/監視をscopeへ追加した。
推論、提出はscope外のまま維持する。

## 背景

exp300では`>3 ft`悪化wellのselection-regret SSEの52.3%が、oracle K16を別候補へ誤rankingしたrowに由来した。
一方、oracle candidate IDやtrue errorはraw-testで使えない。K解像度を変えたときの予測spread・slope spread・
boundary近傍のjump spreadが、K16の信頼度をtarget-freeに表すかだけを0-boosterで検証する。

## 制約

- Route: `ml_model`。後続用途がselector featureだからであり、exp303自身はモデルを学習しない。
- 親: corrected `exp264_exp263_candidate_confidence_dual_selector` Stage C v6。
- 候補入力: exp302でSHA固定された`K=12/16/24`だけ。
- 開始条件: exp302 technical PASS、exp302 candidate novelty PASS、exp276完了、exp276 promotion guard FAILの全て。
- exp276がPASSした場合は、ユーザーが再承認しない限りexp303を実装しない。
- primary unitはexp293と同じoriginの非重複H512 block。
- primary scoreは3成分のouter-train empirical percentile平均、H512内p90に固定する。
- feature/schema/score/blockをtruthなしでSHA freezeしてからtrue suffix TVTを接続する。
- true TVT、error、oracle candidate/rank、bad-well label、well ID ruleをfeatureやscore選択に使わない。
- feature/weight/horizon/threshold/direction grid、selector/Stage D retraining、prediction変更、inference、submissionは禁止。

## 実行量契約

- fixed readout variants: 1
- evaluation folds: 5（学習fold 0）
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- candidate regeneration / parent retraining: `0 / 0`
- GPU: 0、CPUのみ

## 受け入れ基準

- requirements/design/tasklistとexp303 configに、dependency/cancellation、固定feature、primary score、label、
  freeze順序、PASS/FAIL、禁止事項が明記される。
- primary positive labelはH512 blockで`RMSE(K16)+0.25 <= RMSE(exp264 selected hard)`に固定される。
- scientific PASSはpooled AUC `>=0.65`、4/5 foldsでAUC`>0.5`、top/bottom quintileのpositive rate lift
  `>=1.5x`、mean K16 benefit差`>=0.25 ft`、1000+とhidden-like 2面の方向guardをすべて満たす。
- PASSしてもexp303内でgate/selector/predictionを作らず、別のadd-only selector-feature実験の根拠に限定する。
- FAIL時はfeature/score/threshold/horizonを救済せずbranchを閉じる。
- `make validate-exp EXP=exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264`と
  `make validate-template`が通る。
- 別名compact self-contained source/Notebookと専用testを実装し、明示承認後だけ正規Notebookへ採用する。
- Kaggle version 1のreadout生成物だけを作り、prediction、inference、submissionを新規作成しない。
