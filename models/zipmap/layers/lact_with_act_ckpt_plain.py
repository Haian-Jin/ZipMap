"""
Currently, this file implements certain customization for the bidirectional LaCT with SwiGLU fast weight function.
So, it cannot be used for causal LaCT.

"""

import torch
from einops import rearrange
from torch.autograd.function import once_differentiable


@torch.compile(dynamic=True)
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


@torch.compile(dynamic=True)
def ref_pytorch_swiglu_bwd_bwd_fused(
    dh: torch.Tensor,  # [b, d, l]
    x0: torch.Tensor,  # [b, d, l]
    x2: torch.Tensor,  # [b, d, l]
    lr0: torch.Tensor,  # [b, 1, l]
    lr1: torch.Tensor,  # [b, 1, l]
    lr2: torch.Tensor,  # [b, 1, l]
    grad_dx0: torch.Tensor,  # [b, d, l]
    grad_dx2: torch.Tensor,  # [b, d, l]
    grad_hidden_lr1: torch.Tensor,  # [b, d, l]
):
    """
    In previous fwd pass:
    dx0 = lr0 * dh * x2 * sigma * (1 + x0 * (1 - sigma))
    dx2 = lr2 * dh * silu(x0)
    hidden_lr1 = lr1 * x2 * silu(x0)

    In this backward pass:
    grad_dh = grad_dx0 * lr0 * x2 * sigma * (1 + x0 * (1 - sigma)) + grad_dx2 * lr2 * silu(x0)

    grad_x2 = grad_dx0 * lr0 * dh * sigma * (1 + x0 * (1 - sigma)) + grad_hidden_lr1 * lr1 * sigma * x0
    # for grad_x0, a little bit tricky,
    - grad_sigma = grad_dx0 * lr0 * dh * x2 * (1 + x0 - 2 sigma * x0)
    - grad_x0_naive  = grad_dx2 * lr2 * dh * sigma * (1 + x0 * (1 - sigma)) +  grad_dx0 * lr0 * dh * x2 * sigma * (1 - sigma) + grad_hidden_lr1 * lr1 * x2 * dsilu_x0_multiplier
    grad_x0 = grad_x0_naive + grad_sigma * sigma * (1 - sigma)

    # then sum of the last dimension (the d dimension!)
    grad_lr0 = grad_dx0 * dh * x2 * sigma * (1 + x0 * (1 - sigma)) # need to sum over all the d of the same l
    grad_lr2 = grad_dx2 * dh * silu(x0)
    grad_lr1 = grad_hidden_lr1 * x2 * sigma * x0

    """
    # lr0 = lr0.unsqueeze(dim=1)
    # lr1 = lr1.unsqueeze(dim=1)
    # lr2 = lr2.unsqueeze(dim=1)

    sigma = torch.sigmoid(x0)
    silu_x0 = torch.nn.functional.silu(x0)
    silu_bp_multiplier = sigma * (1 + x0 * (1 - sigma))
    grad_dh = grad_dx0 * lr0 * x2 * silu_bp_multiplier + grad_dx2 * lr2 * silu_x0
    grad_x2 = grad_dx0 * lr0 * dh * silu_bp_multiplier + grad_hidden_lr1 * lr1 * silu_x0

    grad_sigma = grad_dx0 * lr0 * dh * x2 * (1 + x0 - 2 * sigma * x0)
    grad_x0_naive = (
        grad_dx2 * lr2 * dh + grad_hidden_lr1 * lr1 * x2
    ) * silu_bp_multiplier + grad_dx0 * lr0 * dh * x2 * sigma * (1 - sigma)
    grad_x0 = grad_x0_naive + grad_sigma * sigma * (1 - sigma)
    grad_lr0 = grad_dx0 * dh * x2 * silu_bp_multiplier
    grad_lr1 = grad_hidden_lr1 * x2 * silu_x0
    grad_lr2 = grad_dx2 * dh * silu_x0

    grad_lr0 = grad_lr0.sum(dim=1, keepdim=True)
    grad_lr1 = grad_lr1.sum(dim=1, keepdim=True)
    grad_lr2 = grad_lr2.sum(dim=1, keepdim=True)

    # also for the first order backward:

    dx2 = silu_x0 * dh
    dx0 = dh * x2 * silu_bp_multiplier

    dx0 = dx0 * lr0
    dx2 = dx2 * lr2

    hidden_lr1 = lr1 * x2 * silu_x0

    return grad_dh, grad_x0, grad_x2, grad_lr0, grad_lr1, grad_lr2, dx0, dx2, hidden_lr1


class ComputeLactFastWeightGrads(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W0, W1, W2, K, V, lr0, lr1, lr2):
        """
        Args:
            W0, W2: [B, M, K] or [B, Hidden, D]
            W1:     [B, K, M] or [B, D, Hidden]
            K, V:      [B, N, K] or [B, num_Tokens, D]
            lr0, lr1, lr2:    [B, num_tokens]
        outs:

            DW0: [B, Hidden, D]
            DW1: [B, D, Hidden]
            DW2: [B, Hidden, D]
        """

        Y0 = torch.bmm(W0, K.transpose(1, 2))
        Y2 = torch.bmm(W2, K.transpose(1, 2))
        DHidden = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))

        ### Element-wise ops

        x0_sigmoid = torch.sigmoid(Y0)

        dx2 = x0_sigmoid * Y0 * DHidden
        dx0 = DHidden * Y2 * x0_sigmoid * (1 + Y0 * (1 - x0_sigmoid))

        DY0_with_lr0 = dx0 * lr0
        DY2_with_lr2 = dx2 * lr2

        Hidden_with_lr1 = lr1 * Y2 * torch.nn.functional.silu(Y0)
        ### Element-wise ops done.

        DW0 = torch.bmm(DY0_with_lr0, K)
        DW2 = torch.bmm(DY2_with_lr2, K)
        DW1 = torch.bmm(V.transpose(1, 2), Hidden_with_lr1.transpose(1, 2))

        # note Y0, Y2 will be recomputed in the backward pass.
        ctx.save_for_backward(W0, W1, W2, K, V, lr0, lr1, lr2)
        return DW0, DW1, DW2

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dw0, grad_dw1, grad_dw2):
        """
        Args:
            grad_dw0: [B, Hidden, D]
            grad_dw1: [B, D, Hidden]
            grad_dw2: [B, Hidden, D]
        Outs:
            grad_W0: [B, Hidden, D]
            grad_W1: [B, D, Hidden]
            grad_W2: [B, Hidden, D]
            grad_K: [B, num_tokens, D]
            grad_V: [B, num_tokens, D]
            grad_lr0: [B, num_tokens]
            grad_lr1: [B, num_tokens]
            grad_lr2: [B, num_tokens]

        Total FLOPS: 24 * B * Hidden * D * num_tokens + 6 * B * Hidden * D * num_tokens
        # 24 for backward matmuls, and 6 for forward recomputation.
        """
        W0, W1, W2, K, V, lr0, lr1, lr2 = ctx.saved_tensors

        Y0 = torch.bmm(W0, K.transpose(1, 2))
        Y2 = torch.bmm(W2, K.transpose(1, 2))
        DHidden = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))

        grad_Hidden_with_lr1 = torch.bmm(grad_dw1.transpose(1, 2), V.transpose(1, 2))
        grad_DY0_with_lr0 = torch.bmm(grad_dw0, K.transpose(1, 2))
        grad_DY2_with_lr2 = torch.bmm(grad_dw2, K.transpose(1, 2))

        (
            grad_DHidden,
            grad_Y0,
            grad_Y2,
            grad_lr0,
            grad_lr1,
            grad_lr2,
            DY0_with_lr0,
            DY2_with_lr2,
            Hidden_with_lr1,
        ) = ref_pytorch_swiglu_bwd_bwd_fused(
            DHidden,
            Y0,
            Y2,
            lr0,
            lr1,
            lr2,
            grad_DY0_with_lr0,
            grad_DY2_with_lr2,
            grad_Hidden_with_lr1,
        )

        grad_K = torch.bmm(DY0_with_lr0.transpose(1, 2), grad_dw0) + torch.bmm(
            DY2_with_lr2.transpose(1, 2), grad_dw2
        )
        grad_K = (
            grad_K
            + torch.bmm(grad_Y0.transpose(1, 2), W0)
            + torch.bmm(grad_Y2.transpose(1, 2), W2)
        )

        grad_V = torch.bmm(
            grad_DHidden.transpose(1, 2), W1.transpose(1, 2)
        ) + torch.bmm(Hidden_with_lr1.transpose(1, 2), grad_dw1.transpose(1, 2))

        grad_W1 = torch.bmm(V.transpose(1, 2), grad_DHidden.transpose(1, 2))
        grad_W0 = torch.bmm(grad_Y0, K)
        grad_W2 = torch.bmm(grad_Y2, K)

        return (
            grad_W0,
            grad_W1,
            grad_W2,
            grad_K,
            grad_V,
            grad_lr0,
            grad_lr1,
            grad_lr2,
        )


lact_swiglu_ffn_fast_weight_grads_with_ckpt = ComputeLactFastWeightGrads.apply


class SwiGLUFFNFwdWithCkpt(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(
        cast_inputs=torch.bfloat16, device_type="cuda"
    )  # let autocast cast once
    def forward(ctx, W0, W1, W2, X):
        """
        Args:
            W0, W2: [B, M, K] or [B, Hidden, D]
            W1:     [B, K, M] or [B, D, Hidden]
            X:      [M, N, K] or [B, num_Tokens, D]
        Outs:
            Hidden: [B, N, K] or [B, num_tokens, Hidden]

        W1 @ [SiLU(W0 @ X.T) * (W2 @ X.T)]
        """

        silu_y0 = torch.nn.functional.silu(
            torch.bmm(W0, X.transpose(1, 2)), inplace=True
        )

        Hidden = torch.bmm(W2, X.transpose(1, 2)) * silu_y0

        output = torch.bmm(Hidden.transpose(1, 2), W1.transpose(1, 2))

        ctx.save_for_backward(W0, W1, W2, X)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: [B, num_tokens, D]
        Outs:
            grad_W0: [B, Hidden, D]
            grad_W1: [B, D, Hidden]
            grad_W2: [B, Hidden, D]
            grad_X: [B, D, num_tokens]
        """

        W0, W1, W2, X = ctx.saved_tensors
        # [B, Hidden, num_tokens]
        Y0 = torch.bmm(W0, X.transpose(1, 2))
        Y2 = torch.bmm(W2, X.transpose(1, 2))

        DHidden = torch.bmm(W1.transpose(1, 2), grad_out.transpose(1, 2))

        Y0_sigmoid = torch.sigmoid(Y0)
        Hidden = Y2 * Y0_sigmoid * Y0

        DY2 = Y0_sigmoid * Y0 * DHidden
        DY0 = DHidden * Y2 * Y0_sigmoid * (1 + Y0 * (1 - Y0_sigmoid))

        grad_W1 = torch.bmm(grad_out.transpose(1, 2), Hidden.transpose(1, 2))

        # [B, Hidden, num_tokens] @ [B, num_tokens, D] -> [B, Hidden, D]
        grad_W0 = torch.bmm(DY0, X)
        grad_W2 = torch.bmm(DY2, X)

        grad_X = torch.bmm(DY0.transpose(1, 2), W0) + torch.bmm(DY2.transpose(1, 2), W2)

        return (grad_W0, grad_W1, grad_W2, grad_X)


fused_swiglu_ffn_fwd_with_ckpt = SwiGLUFFNFwdWithCkpt.apply


def reference_lact_swiglu_ffn_fast_weight_grads(
    W0, W1, W2, K, V, lr0, lr1, lr2, BatchSize
):
    """
    Args:
        W0, W2: [1, M, K] or [1, Hidden, D]
        W1:     [1, K, M] or [1, D, Hidden]
        K, V:      [1, N, K] or [1, B * num_Tokens, D]
        lr0, lr1, lr2:    [1, B * num_tokens]
    Outs:
        DW0, DW2: [BatchSize, Hidden, D]
        DW1: [BatchSize, D, Hidden]
    Total FLOPS: 12 * BatchSize * Hidden * D * num_tokens
    """

    Y0 = torch.bmm(W0, K.transpose(1, 2))
    Y2 = torch.bmm(W2, K.transpose(1, 2))

    DHidden = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))

    # DY0_with_lr0, DY2_with_lr2, Hidden_with_lr1 = ref_pytorch_swiglu_bwd(
    #     DHidden, Y0, Y2, lr0, lr1, lr2
    # )
    ### Element-wise ops
    Y0 = rearrange(Y0, "one d (b n) ->(one b) d n", b=BatchSize)
    Y2 = rearrange(Y2, "one d (b n) ->(one b) d n", b=BatchSize)
    DHidden = rearrange(DHidden, "one d (b n) ->(one b) d n", b=BatchSize)
    V = rearrange(V, "one (b n) d ->(one b) n d", b=BatchSize)
    K = rearrange(K, "one (b n) d ->(one b) n d", b=BatchSize)
    lr0 = rearrange(lr0, "one (b n) ->(one b) 1 n", b=BatchSize)
    lr1 = rearrange(lr1, "one (b n) ->(one b) 1 n", b=BatchSize)
    lr2 = rearrange(lr2, "one (b n) ->(one b) 1 n", b=BatchSize)

    x0_sigmoid = torch.sigmoid(Y0)

    dx2 = x0_sigmoid * Y0 * DHidden
    dx0 = DHidden * Y2 * x0_sigmoid * (1 + Y0 * (1 - x0_sigmoid))

    DY0_with_lr0 = dx0 * lr0
    DY2_with_lr2 = dx2 * lr2

    Hidden_with_lr1 = lr1 * Y2 * torch.nn.functional.silu(Y0)
    ### Element-wise ops done.

    DW0 = torch.bmm(DY0_with_lr0, K)
    DW2 = torch.bmm(DY2_with_lr2, K)
    DW1 = torch.bmm(V.transpose(1, 2), Hidden_with_lr1.transpose(1, 2))

    return DW0, DW1, DW2
