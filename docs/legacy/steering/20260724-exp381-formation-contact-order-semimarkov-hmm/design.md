# 設計

## アプローチ

outer-trainだけで6地層面を位置へ外挿し、各対象坑井軌跡上の `Z - predicted_F_f = 0` 交差を求める。surfaceは既存のformation系readoutと同じwell-median `FormationPlaneKNN(k=10)`に固定し、outer-train formation中央値だけのconstant surfaceを対照とする。各formationについてMD昇順で最初の有限なexact-zeroまたは符号交差を採用し、隣接2行を線形補間する。multiple crossingのoracle選択や順序制約によるpost-hoc修復はしない。

outer-trainの真surfaceで得た接触からformation別contact-TVT中央値を学ぶ。対象坑井では各known-prefix行・各formationの `TVT_input + Z - predicted_F_f - contact_center_f` を集め、その有限中央値を坑井単一additive offsetとする。予測contact-TVTは `contact_center_f + offset`。prefixが不正、2接触未満、または同一formationの予測/真接触を対応付けられない坑井はeligibleに数えない。

target-free freeze前はouter-validから `MD, X, Y, Z, TVT_input` だけを読む。surface、予測crossing、constant baseline、contact-TVT、resource readoutをcontent SHAでfreezeしてから、outer-validの真Formation/TVTをlate joinし、同じfirst-crossing規則で評価する。

Stage 0ではHMMを使わず、交差位置、contact-TVT、接触順序がouter-validへ予測可能かを測る。合格時のみ、6接触面で区切った7 ordered interval stateを持つsemi-Markov HMMを作る。遷移はstayまたは次の1 stateへのadvanceだけで、skipは禁止する。duration sigmaはouter-train surface dispersionから推定して64〜512 ftへclipし、soft potentialのscaleを0.10、二乗誤差clipを25に固定する。

## 実験範囲

- 対象実験: `exp381_formation_contact_order_semimarkov_hmm`
- Route: `pf_beam`
- 親実験: `exp209_emission_dynamics_direct_hmm`
- 変更する変数: ordered interval state、stay/advance transition、contact由来duration potential。
- 固定する変数: exp209 emission/state grid/prefix、fold、評価scope。
- Stage 0: 0 HMMのcontact predictability readoutと16坑井resource audit。
- Stage 1: 773坑井HMM。Stage 0合格後に別承認。

## Stage 0固定契約

- Fold identity: exp209系が比較に使う保存済みexp226 outer 5-fold OOFの`well_id/fold/row_idx/suffix_offset`だけをpre-freezeで読む。
- 主surface: outer-train wellごとのmedian `X/Y/F_f`を参照点にした`FormationPlaneKNN(k=10)`。
- 対照surface: outer-train well-median `F_f`のformation別global median。
- crossing: full-well、MD昇順のfirst crossing、exact-zero優先、隣接区間は線形補間、接触点の同一formation対応だけを評価。
- eligible well: plane/constant/trueで2つ以上の同一formation接触を持ち、known-prefix単一offsetが有限。
- order: 固定formation順に対するplane crossingと真crossingのstrictly increasing判定をwell単位でANDする。
- fold positive: constant crossing MD MAEからplane crossing MD MAEへの改善がstrict positive。
- resource audit: SHA256順で事前固定した16坑井のsurface/crossing生成時間とRSSをreport-onlyで保存する。
- Stage 0実行量: diagnostic 1、reporting surface 6、outer fold 5、fitted model/HMM/PF/Beam/LightGBM booster 0、parent control再実行0、GPU 0。

## 停止条件

eligible率、event数、接触誤差、順序率、fold安定性のいずれかがgate未達ならStage 1を実装しない。formationの一部をpost-hoc除外する救済は別実験とする。

## 再現性設計

- seed policy: 乱数なしの幾何計算とexact HMM。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: Stage 1でexact/semi-Markov HMM、PFなし。
- 並列処理と乱数の関係: well・formation・state順を固定する。
- CPU/GPU runtime と deterministic flags: CPUのみ。
- train cache / test feature regeneration の SHA 記録方針: fold、surface fit、crossing table、contact target、duration priorをcontent SHAで保存する。
- model manifest / prediction / submission SHA 記録方針: HMM state/transition manifestとprediction SHAを保存、submissionは対象外。
- Kaggle package bootstrap 確認方針: Stage 0実装時にoffline importを検証する。

## リスク

- リークリスク: target生Formation列とfull target TVTの使用をread guardで禁止する。
- CV/LB不一致リスク: 地層面が坑井軌跡と交差しない、順序が局所的に逆転する可能性をeligible/order gateで先に測る。
- ランタイム/メモリリスク: Stage 0で交差探索と16坑井HMMの実測を行い、full run前に止める。
- 再現性リスク: root findingのtie/端点規則、surface fallback、state境界を固定する。
