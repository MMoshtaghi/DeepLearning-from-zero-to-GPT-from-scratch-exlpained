import torch
import torch.nn.functional as F 

@torch.no_grad()
def helper_eval_gpt_ce_losses(model, get_batch, n_class, eval_iter, batch_args):
    out = {}
    for stage in ['train', 'eval']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            xb, yb = get_batch(stage=stage, **batch_args)
        
            logits = model(xb)
            mini_batch_loss = F.cross_entropy(input=logits.view(-1, n_class), target=yb.view(-1))
            
            losses[k] = mini_batch_loss.item()
        out[stage] = losses.mean()
    return out