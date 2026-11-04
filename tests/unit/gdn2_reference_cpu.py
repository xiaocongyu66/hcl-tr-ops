"""GDN-2 状态转移的独立参考实现 + 数值对比（CPU fp32，不依赖 GPU/fla）。

存在理由（计划书 §10 红线）："系统未过单元测试不得进入 10B run"。
此前 `test_gdn2_matches_official_kernel` 因无 fla 而 skip —— **skip 不等于通过**。
这里改用**完全独立的朴素实现**做对照：直接按论文公式用显式矩阵乘法（单位阵、
外积）写一遍，不使用被验证代码共享的任何 einsum/辅助函数，从而能真正暴露
实现里的索引/广播错误。

论文公式（vendor/gdn2/lit_gpt/gdn2.py:57-62 docstring）：
    S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t ⊙ v_t)^T
"""

from __future__ import annotations

import torch


def naive_gdn2_reference(q, k, v, g, b, w, l2norm_qk: bool = True,
                         divide_by_sqrt_t: bool = True):
    """纯矩阵形式的参考实现（慢，但直白）。

    q,k,b,g: [T,H,dk]；v,w: [T,H,dv]
    返回 o: [T,H,dv]
    """
    T, H, dk = k.shape
    dv = v.shape[-1]

    def norm(x):
        n = x.norm(dim=-1, keepdim=True)
        return x / n.clamp_min(1e-12)

    kk = norm(k.double()) if l2norm_qk else k.double()
    qq = norm(q.double()) if l2norm_qk else q.double()
    vv, gg, bb, ww = v.double(), g.double(), b.double(), w.double()

    S = torch.zeros(H, dk, dv, dtype=torch.float64)
    outs = []
    for t in range(T):
        kt, vt, gt, bt, wt, qt = kk[t], vv[t], gg[t], bb[t], ww[t], qq[t]
        for h in range(H):
            Diag = torch.diag(torch.exp(gt[h]))                 # [dk,dk]
            # (I - k (b⊙k)^T)
            bk = (bt[h] * kt[h]).unsqueeze(1)                    # [dk,1]
            M = torch.eye(dk, dtype=torch.float64) - kt[h].unsqueeze(1) * bk.transpose(0, 1)
            S[h] = M @ Diag @ S[h] + kt[h].unsqueeze(1) * (wt[h] * vt[h]).unsqueeze(0)
        outs.append(torch.einsum("hk,hkv->hv", qt, S))
    o = torch.stack(outs, dim=0)                                # [T,H,dv]
    if divide_by_sqrt_t:
        denom = torch.sqrt(torch.arange(1, T + 1, dtype=torch.float64)).view(T, 1, 1)
        o = o / denom
    return o
