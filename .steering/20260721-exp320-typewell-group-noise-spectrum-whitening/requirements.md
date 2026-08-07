# 要件

## 依頼

同一Type Well群に共通するGR residualの自己相関をouter-train truthから推定し、GR値を平滑化せず候補alignment likelihoodをwhitenできるか監査する。設計のみで実装しない。

## 制約

- Route: `pf_beam`。exp311/313 PASSが先行条件。
- primaryはcontiguous finite runから推定するshrunk AR(1)のみ。
- lag/order/clip/shrinkageは固定し、GR smoothingやdecoder変更は禁止する。
- outer-valid truthはAR priorとcandidate scoreを凍結した後だけ読む。

## 受け入れ基準

- group AR(1)、global AR(1)、unpooled well AR(1)、group shuffleを比較する。
- support<64はrho=0、rho clipは[-0.8,0.8]、support k=200とする。
- MRR +0.02、top3 +0.03、4/5 folds、shuffle差 +0.02、hidden-like非悪化、worst +0.25 ft以下を要求する。
- direct TVT prediction、HMM/PF/Beam decode、inference、submissionは行わない。
