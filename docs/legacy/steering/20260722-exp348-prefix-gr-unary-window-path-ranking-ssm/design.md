# 設計

## アプローチ

exp332はwindow内の全legal stateをnormal / label-conditioned forward-backwardへ通し、`log Z(u) - log Z(u + ell_y)`を学習した。本案はwindow scheduleとteacher boundaryを維持しつつ、各windowを代表する有限本のlegal pathだけを採点する。

positive pathはtraining truthをsigma`0.35 ft`のGaussian label emissionへ変換し、fixed exp209 grammarでlabel-conditioned Viterbiを1回だけ実行して得る。negative bankは、positive pathから事前定義したposition/rate perturbationで作る14本と、保存済みexp209 path、geometry-only pathの2本で構成する。bankはmodel fit前にfreezeし、学習中のmodel scoreやouter-valid errorによるhard-negative miningを行わない。

各pathのscoreは、valid rowのneural unary logitとfixed transition/boundary log potentialの合計をrow数で割る。positive scoreを各negative scoreよりmargin`0.05`以上高くするpairwise softplus lossを平均し、exp332と同じlocal CE `0.25`を加える。評価時はranking候補から直接選ばず、学習済みunaryをofficial suffix全体のfixed exact SSMへ渡してposterior meanを得る。

## 2026-07-24 実装決定

- exp347は2026-07-23にStage 0 posterior parity gateをFAILしてterminal closeしたため、exp348の先行条件は成立した。
- exp332 compact self-contained trainの13章構成を維持し、train候補では正例/負例生成、legality、dedup、固定potential事前計算、ranking loss、Stage 0/Stage A orchestrationをセル上で追えるようにする。
- negativeのposition/rate templateは、clipせずgrid外を除外した後、fixed exp209 grammarのjoint Viterbiで最寄りlegal pathへ決定論的に投影する。投影時のposition sigmaは`0.35 ft`、rate-index sigmaは`0.25`に固定してconfigへ置く。
- path scoreのfixed transition/boundary potentialはpath bank freeze時に一度だけ計算する。fit loopはneural unaryのgather、事前計算済みfixed potentialの加算、全negativeのsoftplus平均だけを行い、exact partition sweepを呼ばない。
- Stage 0の固定16 windowsは全件のpath bank、forward-only、full-well controlsを計測し、各suffix quartileの3件、合計12件だけをtemporary optimizerへ使う。各quartileの残り1件、合計4件をearly holdoutとしてtop-1とmargin gateに使う。
- path bank manifestはwindow/path単位のposition/rate content SHA、fixed potential、dedup/exclusion reason、aggregate decompressed content SHAを保存する。数千万rowのstate展開は保存せず、immutable in-memory pathとcontent SHAでfreezeを証明する。

## Path bank contract

### Positive 1本

- fixed exp209 grammar + Gaussian label emission sigma`0.35 ft`のlabel-conditioned Viterbi。
- interior teacher boundaryはexp332と同じtraining-loss-only initial position/rate prior。
- encoderへtruth path、boundary truth、future TVTを渡さない。

### Negative 最大16本

- position grid-index offsets: `[-57, -29, -14, +14, +29, +57]`（約`-19.95, -10.15, -4.90, +4.90, +10.15, +19.95 ft`）。
- constant rate-index offsets: `[-2, -1, +1, +2]`。
- midpointから32 rowsのrate pulse: `[-4, -2, +2, +4]`。
- saved exp209 official-prefix path: 1。
- geometry-only fixed-grammar path: 1。
- position/rate grid外はclipせずinvalid negativeとして除外する。positiveとstate sequenceが同一のpathも除外する。
- unique negativeが12本未満のwindowはzero-lossにせずfail-closedする。
- negative family、順序、path content、除外理由をmanifestへ保存する。

## Loss contract

`score(path) = mean_valid_rows(unary_theta(row, position_state) + transition_logp + boundary_logp)`

`L_rank = mean_neg softplus(0.05 + score(neg) - score(pos))`

`L = 1.0 * L_rank + 0.25 * L_local_CE`

- transition/state grammarは学習しない。
- scoreはpath総和ではなくvalid row平均とし、短いsuffixがmarginを変えないようにする。
- all-negative平均を使い、top-k miningやmodel-dependent weightingを行わない。
- path score計算はgather中心で、normal/label-conditioned partition function、posterior、4 DP sweepsをtraining loopに含めない。

## 固定する科学契約

- window schedule、boundary、architecture、preprocessing、optimizer、8 epochs、AMP、gradient clip、foldはexp332と同じ。
- batchはexp332と同じ1 well-window、gradient accumulation 4とし、exp347のbatched DP変更を混ぜない。
- full-well評価はfixed exact SSM posterior mean。real GR、circular shuffle、geometry-only、保存済みexp209/exp221を比較する。
- ranking candidate pathそのものをvalidation/test予測として選ばない。

## Stage 0

- suffix-length quartile×4件の固定16 windowsでpositive/negative generation、path gather score、ranking forward/backward/optimizer、early-stop forward-only、full-well 3-control decodeを計測する。
- technical gate: manifest row/window identity、positive 1、unique negative`>=12`、finite score/loss/gradient、outer-valid truth access 0、path SHA一致。
- learning gate: fixed early-holdout windowsでpositive top-1 rate`>=0.80`、mean `score(pos)-max score(neg) >=0.02`。
- compute gate: p10-throughput保守的fold外挿`<=8.5 h`、peak`<=14 GB`。
- Stage 0量: active variant 1、temporary model 1、persisted model 0、trained fold 0、LightGBM config/booster/PF/Beam/control再学習各0。
- FAIL時はmargin、negative family/count、path score、window、local CE、architecture、decoder、epochを変更せずcloseする。

## Stage A/B/C

Stage 0全PASSと別承認後だけfold 0の1 neural modelを学習する。ranking指標に加え、freeze後のfull-well real exact decodeがgeometry-only、shuffle、保存済みexp209へ勝ち、p95/worst safetyを満たすことを必須とする。exp331が示した「GR signalはあるがglobal TVT pathが悪い」状態を再発した場合は閉じる。Stage B/Cはexp332と同じ5-fold promotion gateと別承認を要求する。

## 実験範囲

- 対象実験: `exp348_prefix_gr_unary_window_path_ranking_ssm`
- Route: `ensemble`
- 親実験: `exp332_prefix_gr_unary_fixed_window_structured_ssm`
- 変更する変数: exact structured NLLを固定path bankのpairwise softplus rankingへ置換する。
- 固定する変数: data、fold、window schedule、teacher boundary、architecture、local CE、optimizer、state grammar、full-well decoder、controls、promotion gate。
- 優先順位: exp347より後のP3 high-risk branch。exp347と同時実装・同時GPU比較しない。

## 再現性設計

- seed policy: seed 42 + stable SHA256 window schedule/path family/order。
- stochastic処理: CUDA convolution、AdamW、dropout、dataloader order。path bank生成はdeterministicでglobal RNGを使わない。
- PF/Beam / likelihood-PF / LightGBM: 0。
- GPU runtime: Kaggle T4、internet off、worker 0、CuDNN benchmark false、deterministic algorithms warn-only。
- SHA: input/fold/window/boundary/positive/negative/dedup manifest、path bank decompressed content、model、unary/posterior/prediction、package/kernelを記録する。
- deterministic anchor: false。GPU学習とAMPのrerun監査なしにbitwise再現性を主張しない。

## リスク

- 固定negative bank外の誤経路を学習できず、full-well decodeで未知のmode slipが残る可能性がある。
- synthetic negativesが簡単すぎるとranking gateだけ通り、exp331同様にTVT RMSEが悪化し得る。
- transition log potentialが固定のため、モデルが主にposition unary差だけを学ぶ可能性がある。
- truth由来positive/negativeのtraining-only境界を誤るとleakageになる。
- ranking scoreはcalibrated posteriorではないため、full exact decodeとのtrain/eval objective mismatchがexp332より大きい。

## 次のアクション

別承認時だけcompact train候補を正規Notebookへ採用し、Kaggle T4固定16-window Stage 0でtechnical / early-holdout learning / runtime / memoryのAND gateを評価する。全gate PASS前にStage Aへ進めない。
