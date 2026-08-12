# 設計

## アプローチ

exp238 OOF selector-confidence notebookのbase row contractと入力resolverを縮約して再利用する。
exp072 train feature cacheから`id`、`well`、truth、保存済み`likpf_mean`を読み、exp238 final
OOFの`lgb_mean_pred_tvt`を同順で結合する。

PF本体はexp072 v2の`_pf_lik_allseeds`に必要な関数だけをself-contained notebookへ抽出する。
raw train horizontal/typewellをwell単位で読み、`stable_seed("likpf", "train", well_id)`から
128 seed trajectoryを再生する。複数wellをthread batchで生成し、plotはmain threadで順次作る。
各plotでは128本を同色・低alpha・細線で重ね、truthを黒、exp238 OOFをroseの太線で前面に置く。

全128軌跡を全well分保存するとfloat32だけでも約1.94 GBになるため、軌跡はwell単位の一時配列に
限定する。保存するのはPNG、well manifest、plots zip、summary JSONとする。

## 実験範囲

- 対象実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- Route: `ml_model`
- 親実験: exp238 final train v5、PF replay contractはexp072 v2
- 変更する変数: diagnostic plotだけ。likelihood-PF 128 seed trajectoryを可視化する。
- 固定する変数: exp238 OOF、exp072 PF dynamics、500 particles、128 seeds、raw train、final prediction、submission。

## 再現性設計

- seed policy: SHA256の`stable_seed("likpf", "train", well_id)`をwellごとの`seed_base`とし、seed index 0..127を加算する。
- stochastic 処理の有無: likelihood-PF内部にのみある。Numba kernel内で各seed開始時に明示的に`np.random.seed(seed_base + seed_index)`する。
- PF/Beam / likelihood-PF / seed baggingの有無: exp072互換likelihood-PFを500 particles × 128 seedsで再生成する。Beamや他PFは生成しない。
- 並列処理と乱数の関係: `joblib` thread並列はwell単位。各kernel callがwell固有seed baseを持ち、thread schedulingでseed系列を共有しない。
- CPU/GPU runtimeとdeterministic flags: CPU、internet disabled。GPU、model fit、boosterは0。
- train cache / test feature regenerationのSHA記録方針: exp072 cacheとexp238 OOFのdecompressed SHAをsummaryに保存し、regenerated seed meanとsaved meanのmax/mean abs差を保存する。test regenerationはない。
- model manifest / prediction / submission SHA記録方針: 保存済みexp238 OOF SHAだけを入力証拠として記録する。model、test prediction、submissionは生成しない。
- Kaggle package bootstrap確認方針: prepare scriptのzip bootstrapでexp238 `config.yaml`を展開し、package notebookとcanonical notebookのcell source、metadataのCPU/internet-off/run-on-push-false、2 kernel sourcesを検証する。

## リスク

- リークリスク: true TVTはplotとRMSE表示だけに使い、PF state updateやseed選択へ渡さない。
- CV/LB不一致リスク: PF 128本はexp238 final predictionではなく、OOFのdiagnostic overlayに限定する。anchor・提出判断は更新しない。
- ランタイム/メモリリスク: 全773 wellsの128 × 500 PF再生は数時間規模。batch thread並列とwell単位解放を使い、全軌跡の永続化を避ける。
- 再現性リスク: Numba/RNG runtime差をsaved exp072 `likpf_mean` parityでfail-fast監査する。parityが成立しない実行は可視化を正規結果として扱わない。
