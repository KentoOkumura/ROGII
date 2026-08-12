# 要件

## 2026-08-08 修正契約（v2、現行）

- ユーザー指摘により、v1の`1 PF direct output`はstage 2-2再現ではなく、契約不一致の失敗として保持する。実験名や過去提出を別名へ変えて成功扱いしない。
- 同じ`exp517_stage22_pf1_tw_fixedlag192_late_submit`内で、stage 2-2 systemの実装へ修正する。
- input: 公開Ravaghi tabular feature frameと、公開6位解法configのoriginal Optuna group先頭5本`pf_1 / pf_2 / pf_3 / r0_seed32 / r1_seed32`をそれぞれ`twGR`へ適用したtrajectory。各bankは32 seeds、公開particle数、fixed-lag 192、GR-only、anchorなし、learned emissionなしとする。
- target / objective: row target `TVT - last_known_tvt`をtabular regressionで推定する。
- output: 5 PF trajectory由来のrow featureを含むtabular residual prediction。
- loss: LightGBM / CatBoostのRMSE。公開GroupKFold 5分割を使う。
- decode: 公開tabular stackの`3 LightGBM + 2 CatBoost`をOOF予測上のpositive Ridgeで融合し、公開postprocess（PF blend、MD-distance fade、well内Savitzky–Golay）を適用して`last_known_tvt`へ加算する。
- context unit: PFは1 wellのknown-prefix終端からhidden suffix全体をforward処理し、各rowは最大192 future rowsをancestry contextとして使う。tabularはrow-level、split unitはwellである。
- PFのv98以後に追加されたStudent-t likelihood、tempering、ps-combo、physical anchor、learned emission、self/nbr/full smoothingはstage 2-2に含めない。
- tabular基盤は、作者が「public notebooksと同じように入力」と記載した時点の公開Ravaghi artifact/schemaと保存済み公開実装を正とする。train artifactをcontrolとして再学習せず、5 PF featureを加えたscientific variantだけを学習する。
- 実行量: scientific variant 1、PF banks 5、representations 1、LightGBM configs 3、CatBoost configs 2、folds 5、base models 25、Ridge folds 5、control rerun 0。
- score gate: published stage 2-2 CV `7.50`との同条件比較を行う。CVが再現しない限り、実装完了・score再現・late submitとは扱わない。
- v1のPublic `7.825` / Private `9.689`は、`pf_1 direct`という別出力の失敗証拠として残し、v2の成果へ付け替えない。

## 依頼と手法契約

- 依頼原文: 6位解法のPF単体をlate submitで再現する。追加確認により、最終`pfA × twGR`ではなくwriteup stage 2-2を目標とし、公開最終`pf_1`設定を使うproxyで進める。
- 期待する成果: stage 2-2の中核として説明された`pf_1 × twGR × fixed-lag 192`だけをcurrent hidden testから生成し、`LATE SUBMIT`と明記した固定版を1回だけ提出する。
- 一次資料: `docs/discussions/rogii-wellbore-geology-prediction-733226.md`のstage 2-1 / 2-2。stage 2-2はGPU PF、32 seeds、fixed-lag particle smoother、lag 192を説明する。
- 参照実装: 公開kernel `k256net/public20th-private6th-pf-pf-pf-pf-and-bagging`、保存source SHA256 `b44f7889d6abdf9b027d33cb6c6b45f23902d609fae8d06f332914017784c924`、公開`v96_art/pf_banks_config.json`の`pf_1`。
- input: runtime horizontal wellのknown-prefix終端とhidden suffix `MD, Z, GR`、対応typewellの`TVT, GR`。self/nbr表現、GR-free anchor、learned emissionは使わない。
- target / objective: 学習target/lossは持たず、hidden suffixのstate `position = TVT + Z`とrateの逐次posteriorをbootstrap particle filterで近似する。
- output: 各hidden rowの`pf_1 × twGR` fixed-lag-192 smoothed TVT mean、seed間std、run log-likelihood。late submissionはsmoothed meanをruntime `sample_submission.csv`へIDで1対1整列する。
- loss: 学習lossなし。各particleの観測weightはtypewell TVT-grid上のGRとhorizontal-well GRの差に基づく公開power likelihood。物理anchor likelihoodとNN-emission likelihoodはstage 2-2より後の機構として使わない。
- decode: 公開最終`pf_1`の600 particlesを32 seedsで実行し、各時点で192-row先までのparticle ancestryをbacktraceするfixed-lag smoother。末尾192-rowは利用可能なsuffix末端まででbacktraceし、seed間は正規化run log-likelihoodでsoft weightingしたmean/stdを得る。追加blend/postprocessなし。
- context unit: `local future window within one hidden suffix`。known-prefix末端からwhole hidden suffixをforward filterし、各rowは最大192 future rowsのancestry contextを使う。
- 実装区分: `proxy`。
- 省略・置換する機構と理由: 公開stage 2-2の5 PF parameter setsと当時のexact configは非公開であるため、図示された`pf_1 × twGR`だけを単体decodeし、parameterは最終公開v96の`pf_1`へ置換する。stage 2-2の5 PF入力を受けるtabular modelもPF単体というユーザー指定により省略する。
- proxyで検証できない主張: stage 2-2掲載値 CV `7.50` / Public `6.724` / Private `7.404`の再現、当時の5 PF candidate setの再現、tabular modelへのsmoother gain `CV -0.3 / LB -0.4`の再現、当時の`pf_1` parameter identity。
- 完全再現に必要な追加物: 作者のstage 2-2時点の5 PF config、tabular feature/schema/modelまたはtraining source、split、当時の予測artifact。GPU時間だけでは欠落契約を復元できない。
- proxy承認: 2026-08-07、上記の「最終公開`pf_1` parameterをfixed-lag 192で動かすproxy」を説明後、ユーザーが「それで進めてください」と明示承認した。

## 制約

- Route: `pf_beam`。tabular modelを含めないため`ensemble`ではない。
- submission phase: `post_competition_late_submission`。Notebook title、submission message、README、notes、resultに`LATE SUBMIT`を明記する。
- scientific variantは`pf_1 × twGR × fixed-lag 192`の1本。600 particles、32 seeds、公開最終`pf_1` config、PF seedを実行前に固定する。
- GR-free anchor training、learned-emission checkpoint、ML model、LightGBM、他PF bank、他GR representation、full smoothingを実行しない。
- runtimeのtrain/testを動的に解決し、公開test固有well ID、well数、row数、prediction、SHAで分岐しない。runtime sample submissionをschema/order/ID集合の正とする。
- `docs/06_reproducibility.md`に従い、source/config、well/row、candidate、prediction、submission SHAとGPU/device/chunk条件を記録する。
- Kaggle push直前にmetadata resourceとGPU quotaを確認する。CLIで取得できないActive Sessions数はgateにしない。GPU同時上限は2、CPUは5。
- technical gate通過後のlate submissionは固定版1回。LB後parameter/seed/lag/postprocess調整はしない。

## 受け入れ基準

- 手法契約の`input / target / output / loss / decode / context unit`がコードと一致する。
- 実験名と文書が`stage 2-2 exact reproduction`や`6th-place full solution`を主張せず、`pf1 tw fixedlag192 proxy`と明記する。
- 公開v96 `pf_1` configの全key/valueとvendor copyのraw SHAを固定し、runtimeで正規化text SHAも検証する。
- late-submit pathが`pf_1`、`tw`、600 particles、32 seeds、`smooth_mode=fixedlag`、`smooth_lag=192`だけを実行する。
- anchor/emission/ML/他bank/他representation/full smootherが実行されないことをcontract testで確認する。
- inferenceがsample IDへ1対1整列し、missing/duplicate/extra/non-finiteを0にする。
- static、Jupytext pairing、strict experiment validation、Kaggle full execution、submit-checkを通す。
- deterministic anchorと呼ぶ場合はprediction/submission SHAとKaggle version、rerun一致を記録する。単一GPU runだけなら`stochastic proxy replay`と記録する。
