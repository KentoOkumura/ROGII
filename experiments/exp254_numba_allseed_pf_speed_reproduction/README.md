# exp254 Numba all-seed PF speed reproduction

`backlog/KAGGLE_DIRECTION.md` の `numba_allseed_pf_speed_reproduction` を実装する、
PF/Beam route の速度・再現性監査です。

## 状態

Kaggle CPU probe v1完了、全parity・決定性guard通過です。ただし実用的な高速化にならず、
cached 300-candidate再集約にも現在の用途がないため、完了・不採用としてbranchを閉じました。
full workload、inference、submissionは実行しません。

## 仮説

公開実装で示唆された「6時間から2–3分」「ensemble約300倍」はPF dynamics自体の変更ではなく、
seed loop全体のJIT化と、seed trajectory / per-seed log-likelihoodを一度だけ計算して多数候補へ
再集約するwarm generationの分離で説明できる可能性があります。

## 検証方針

exp243 v3 で exact parity が確認された exp072 likelihood-PF を固定し、次を分離して測ります。

- Python 側で single-seed Numba kernel を128回呼ぶ legacy seed loop
- seed loop 全体を1回の Numba call に入れる all-seed kernel
- all-seed trajectory / log-likelihood cache を再利用する warm candidate generation

精度実験ではありません。true TVT、error、oracle、selector、ML feature、inference、submissionは
使いません。既定の`probe` modeではeval lengthの10/50/90%分位から決定的に選んだ3 wellsだけを
測り、seed数`1/4/16/32/64/128`、candidate spec数`1/10/100/300`を比較します。

`full_workload` modeは、probe summaryのSHAをconfigへ明示し、全parity・決定性guardが通った場合だけ
773 wellsの固定all-seed + 300 candidate workloadを実測できるfail-closed構成です。

Kaggle CPU Notebookを正とし、GPU・internet・model training・competition submitは無効です。

## 所見

Kaggle実測では128-seed PF coreの3-well合計がlegacy 80.897349秒、all-seed 81.754755秒で、
all-seed化だけの速度改善はありませんでした。一方、保存済みseed bankから300 candidateを作るwarm
generationは3 wells合計0.104562秒で、PF coreに対し706–888倍軽量でした。trajectory、
log-likelihood、mean、ESS、resampling、exp243保存値、repeat/cache SHAはすべてexactです。
ただし通常推論では300集約候補を必要とせず、exp252でもseed候補の選択信号が弱いため採用しません。

773-wellのall-seed + 300-candidate外挿は21,436.315秒（約5時間57分）ですが、これは3 wellsからの
projectionであり実測full runtimeではありません。追加実測の用途がないためfull workloadは閉じています。
