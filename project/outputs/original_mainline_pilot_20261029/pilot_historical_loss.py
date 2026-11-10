"""Historical step-35 loss components with an explicit masked-sequence adaptation.

The caller must supply the clean/type logits, augmented embedding and the
clean encoder sequence; no training loop is defined here.
"""
import torch
from torch.nn import functional as F

def historical_bundle(bin_head, type_head, controller, binary_logits, type_logits,
                      augmented_embedding, clean_sequence, physical_state,
                      mask, labels, label_class, class_weights, retention):
    maskf=mask.to(physical_state.dtype)
    state=(physical_state*maskf.unsqueeze(-1)).sum(1)/maskf.sum(1,keepdim=True).clamp_min(1)
    control=controller(state)
    eps=0.05+0.2*(torch.tanh(control[:,0])+1)/2
    mu=augmented_embedding.mean(0,keepdim=True).detach()
    sd=augmented_embedding.std(0,keepdim=True).detach().clamp_min(1e-5)
    perturbed=mu+sd*(((augmented_embedding-mu)/sd)*(1+eps[:,None])+0.02*torch.tanh(control[:,1])[:,None])
    binary=F.binary_cross_entropy_with_logits(binary_logits,labels)
    positive=labels>0
    type_ce=F.cross_entropy(type_logits[positive],label_class[positive]-1,weight=class_weights) if positive.any() else binary*0
    aug_bce=F.binary_cross_entropy_with_logits(bin_head(perturbed).squeeze(-1),labels)
    p1=F.softmax(type_logits,-1);p2=F.softmax(type_head(perturbed),-1);middle=(p1+p2)/2
    def kl(a,b):return (a*(torch.log(a+1e-8)-torch.log(b+1e-8))).sum(-1)
    js=((kl(p1,middle)+kl(p2,middle))*0.5*retention).mean()
    # clean_sequence is [batch,time,hidden] and reflects actual backbone states.
    delta=clean_sequence[:,2:]-2*clean_sequence[:,1:-1]+clean_sequence[:,:-2]
    valid=mask[:,2:] & mask[:,1:-1] & mask[:,:-2]
    smooth=(delta.square()*valid.unsqueeze(-1)).sum()/(valid.sum()*clean_sequence.shape[-1]).clamp_min(1)
    total=binary+0.5*type_ce+0.2*js+0.3*aug_bce+0.01*smooth
    return total, {'binary':binary,'type_ce':type_ce,'js':js,'aug_bce':aug_bce,'smooth':smooth,'epsilon':eps}
