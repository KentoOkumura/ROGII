# タスクリスト

## 閉鎖

- Stage 0はreal-vs-circular fold gateが2/5でFAIL。exp341へtableを渡さず、救済再実行もしない。

## 完了

- 2026-07-22: exp339として採番し、steeringと実験scaffoldを作成した。
- 2026-07-22: fold、pseudo-gap、bin、shrinkage、controls、gate、SHAを設計固定した。
- 2026-07-22: ユーザー承認に基づきcompact self-contained Stage 0 train候補、fail-closed inference候補、正規Notebook、contract testを実装した。
- 2026-07-22: Jupytext、py_compile、Ruff、exp339 contract test、strict experiment validationを実施した。
- 2026-07-22: Kaggle実行、HMM、推論、提出は行っていない。
- 2026-07-22: CPU Stage 0実行前に、scientific readout 1、control 2、outer audit fold 5、model config / trained fold / booster / HMM / 親control再学習はすべて0と再確認した。
- 2026-07-22: 55文字canonical候補はKaggle SaveKernel 400で未作成。50文字制約に合わせ、意味を保った47文字の`exp339-missing-gap-pseudomask-uncertainty-train`へ短縮して再packageすることを記録した。
- 2026-07-22: 短縮canonical kernel version 1（id_no 128226213）をCPU / internet offで完了した。
- 2026-07-22: global比NLL、coverage、校正、長さ相関は通ったが、real placementのcircular比fold勝利が2/5で固定4/5 gateをFAIL。exp341を依存FAILで閉じた。
