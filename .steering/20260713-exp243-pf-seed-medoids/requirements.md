# 要件

## 仮説

PF seed trajectoryの複数modeを平均せずmedoid候補として残すと、長期pathの候補coverageが増える。

## 依頼

`KAGGLE_DIRECTION.md` の高優先度バックログ `pf_seed_medoids` を実装する。exp072互換
likelihood-PFの128 seed平均だけでなく、seed別trajectoryをtarget-free距離でクラスタリングし、
各modeを代表する実在trajectoryであるmedoidをcandidate pathとして残す。

## 制約

- Route: `pf_beam`。
- PFは500 particles × 128 seeds、raw GR Gaussian likelihood、exp072 transition / resamplingを固定する。
- seedはexp072の`sha256("likpf::train::<well>") % 2147483647 + 1 + seed_index`契約を再現する。
- trajectoryはreplay meanとK-medoids計算までfloat64を維持し、保存列だけfloat32へ変換する。
- typewell TVT/GR、horizontal MD/Z/GR、GR grid、sigma計算を含むPF入力はCSV読込時からfloat64を維持し、float32を経由しない。
- canonical exp072 v2 feature cacheとschemaをdecompressed/raw SHAで固定し、saved `likpf_mean`だけはSHA固定したexp209 enriched cacheから別名復元して一対一結合する。
- 距離は承認済みのtail前半1.0 / 後半1.5 weighted trajectory RMSEに固定する。
- K-medoidsは決定的BUILD+PAM、Kは`3/5/8`をすべて生成し、targetで選択しない。
- medoidは実在seed trajectoryとし、centroidやmedoid平均をdirect predictionにしない。
- true TVT / target / error / oracle / Public LBをclustering、K、medoid、candidate順に使わない。
- LightGBM、fold、booster、親/control再学習、GPU、raw-test inference、submissionは行わない。
- `docs/06_reproducibility.md`に従い、input、row candidate、cluster manifestのSHAを記録する。

## 受け入れ基準

- 128 seed trajectoryをwell単位で保持し、処理後に解放する。
- K=3/5/8についてmedoid候補、cluster mass、likelihood mass、entropy/HHI、within/between距離を保存する。
- 既存`likpf_mean`をfallbackとして保持し、exp237 base8 candidate unionも再構成する。
- row、128/256/512-row block、whole-well oracle、unique-best率を保存する。
- overall、1000+、hidden-like、by-well、worst-wellを比較できる。
- train / inference notebookがJupytext percent形式から生成され、inferenceは明示的にdisabledである。
- `py_compile`、`ruff --select F821`、Jupytext `--test`、`make validate-exp`が通る。
- one-well parity probeではexp072入力とexp243入力のgrid min/max、GR sigma、initial rate、seed baseが一致し、saved `likpf_mean`との差を確認できる。
- Kaggle push前に1 PF replay、3 K-medoids postprocess、LightGBM 0、fold 0、booster 0を記録する。

## 次のアクション

実装・静的検証後は自動pushしない。2026-07-14にユーザーが、過去4 shardのCPU時間合計
約9時間11分を踏まえ、full rerunを1 CPU notebookで実行することを明示承認した。
