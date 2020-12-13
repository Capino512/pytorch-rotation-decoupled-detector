

def adjust_lr_multi_step(optimizer, step, cfg, warm_up=None):
    for param_group in optimizer.param_groups:
        if warm_up is not None and step <= warm_up[0]:
            param_group['lr'] = warm_up[1] + step / warm_up[0] * (warm_up[2] - warm_up[1])
        else:
            for s, lr in cfg:
                if s is None or step <= s:
                    param_group['lr'] = lr
                    break
