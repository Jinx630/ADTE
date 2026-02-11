import torch

from copy import deepcopy
from .base import BaseTTAModule
from clip.custom_clip import get_clip
import torch.nn.functional as F
from utils.tools import print, greedy_break

def confidence_filter(logits: torch.Tensor, probs: torch.Tensor, top:float, scale_w: torch.Tensor, bias: torch.Tensor, use_adte, return_idx: bool=False):
    
    if use_adte:
        batch_entropy = tsallis_entropy_minimization(probs, q=scale_w)
    else:
        batch_entropy = shannon_entropy_minimization(logits, probs, bias)

    full_idx = torch.argsort(batch_entropy, descending=False)
    filt_idx = full_idx[:max(int(batch_entropy.size()[0] * top), 1)]
    if not return_idx:
        return logits[filt_idx]
    return logits[filt_idx], filt_idx, full_idx

def tsallis_entropy_minimization(probs, q=2):
    # probs = F.softmax(logits, dim=1)
    if torch.is_tensor(q):
        N = len(q)
        entropy = torch.sum((torch.pow(probs,q))/(1 - q), dim=1)
    else:
        entropy = (1 - torch.sum(probs ** q, dim=1)) / (q - 1)
    return entropy

def shannon_entropy_minimization(logits, probs, bias):
    # probs = F.softmax(logits - bias, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy

class Zero(BaseTTAModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_views = 64
        self.gamma = 0.1
        self.class_num = len(kwargs['classnames'])

        self.model = get_clip(
            kwargs.get("arch"),
            kwargs.get("pretrained"),
            kwargs.get("gpu"),
            kwargs.get("ctx_init"),
            cache_text_features=True,
            use_text_templates=bool(kwargs.get("use_templates")),
            use_cupl_descriptions=bool(kwargs.get("use_cupl")),
            freeze_text=True,
            freeze_vision=True,
            maple_weights=kwargs.get("maple_weights"),
            maple_seed=kwargs.get("seed")
        )

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print('=> Freezing all parameters.')

        self.name = "ZERO"
        print(f"num_views: {self.num_views}, gamma: {self.gamma}")

        self.q = None
        self.pre_compute_q = kwargs.get("pre_compute_q")
        self.is_debias = True
        self.memory_size = 10
        self.memory_bank = torch.zeros(self.class_num, self.memory_size, self.class_num).to(self.model.device)
        self.logit_gap = torch.full((self.class_num, self.memory_size), -1., dtype=torch.float).to(self.model.device)
        self.start_debias = 100
        self.scale_w = torch.ones(self.class_num)

    def compute_P(self, memory_bank, logit_gap):
        K = memory_bank.shape[0]
        P = torch.zeros(K, K).to(memory_bank.device)
        for j in range(K):
            valid_indices = (logit_gap[j] != -1).nonzero(as_tuple=True)[0]
            valid_samples = memory_bank[j][valid_indices]
            Nj = len(valid_samples)
            if Nj > 0:
                softmax_values = torch.softmax(valid_samples, dim=1)
                P[:, j] = torch.sum(softmax_values, dim=0) / Nj
        return P

    def estimate_q(self, P, epsilon=1e-4, n=500):
        K = P.shape[0]
        q = torch.ones(K).to(P.device) / K
        for _ in range(n):
            q_new = torch.matmul(P, q)
            if torch.norm(q_new - q) < epsilon:
                break
            q = q_new
        return q

    def compute_q(self, l, probs, target = None):
    
        batch_entropy = shannon_entropy_minimization(probs)
        # batch_entropy = tsallis_entropy_minimization(probs)

        full_idx = torch.argsort(l.logsumexp(1), descending=False)
        filt_idx = full_idx[:32]

        origin_output = l.mean(0)
        top_1, top_2 = torch.topk(origin_output, 2)[0]
        logit_diff = top_1 - top_2

        if logit_diff > 1:
            pseudo_label = torch.argmax(origin_output)
            # pseudo_label = target
            min_index = torch.argmin(self.logit_gap[pseudo_label])
            self.memory_bank[pseudo_label][min_index] = origin_output
            self.logit_gap[pseudo_label][min_index] = logit_diff

        P = self.compute_P(self.memory_bank, self.logit_gap)
        q = self.estimate_q(P)

        return torch.log(q)

    @torch.no_grad()
    def zero(self, views, target, is_zero, use_adte=0):

        if hasattr(self, "z_txt"):
            z_txt = self.z_txt
        else:
            z_txt = self.model.get_text_features()
            self.z_txt = z_txt
        
        z_img = self.model.get_image_features(views) 
        logits = z_img @ z_txt.t() * self.model.logit_scale.data.exp()

        if is_zero:
            return logits

        if self.pre_compute_q:
            q = self.q
        else:
            if self.is_debias:
                q = self.compute_q(logits, logits.softmax(1))
                memory_size = (self.logit_gap != -1).sum()
                if memory_size < self.start_debias:
                    q = torch.tensor([0]).to(self.logit_gap.device)
                if memory_size == self.start_debias:
                    print(f"Start Debias, current memory bank size: {(self.logit_gap != -1).sum()}")

        prob = (logits).softmax(1) # probabilities
        logits_filt = confidence_filter(logits, prob, top=self.gamma, scale_w=self.scale_w, bias=q, use_adte=use_adte) # retain most confident views
        
        p_bar = (logits_filt).softmax(1).sum(0) # marginalize
    
        return p_bar.unsqueeze(0)

    def forward(self, images, target, is_zero=0, use_adte=0):
        return self.zero(images, target, is_zero, use_adte)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)