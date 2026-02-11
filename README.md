## Dataset dir:

```
dataset
	|-imagenet-1k
	|-imagenet-a
	|-imagenet-r
	|-imagenet-s
	|-imagenet-v2
	|-cross_domain_dataset...
```

Dataset download and setup can be referred at [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)

## Run:

ImageNet-A, **Zero-shot CLIP**, hand-crafted templates, ViT-B/16:

```
python run.py --set_id=A --gpu=0 --is_zero --arch=ViT-B/16 --templates
```

ImageNet-A, **Shannon Entropy**, hand-crafted templates, ViT-B/16:

```
python run.py --set_id=A --gpu=0 --arch=ViT-B/16 --templates
```

ImageNet-A, **Tsallis Entropy (q=0.1)**, hand-crafted templates, ViT-B/16:

```
python run.py --set_id=A --gpu=0 --same_q --scale_w=0.1 --arch=ViT-B/16 --templates
```

ImageNet-A, **Adaptive Debiasing Tsallis Entropy**, hand-crafted templates, ViT-B/16:

```
python run.py --set_id=A --gpu=0 --use_adte --arch=ViT-B/16 --templates --q_path=bias_pt/imagenet_A_q_1_10_ensemble_B16.pt
```

## Code:

Shannon Entropy

```
# Shannon Entropy
def shannon_entropy_minimization(logits, probs, bias):
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy
```

**Tsallis Entropy** and **Adaptive Debiasing Tsallis Entropy**

```
def tsallis_entropy_minimization(probs, q=1):
    if torch.is_tensor(q):
        # Adaptive Debiasing Tsallis Entropy
        entropy = torch.sum((torch.pow(probs,q))/(1 - q), dim=1)
    else:
        # Tsallis Entropy
        entropy = (1 - torch.sum(probs ** q, dim=1)) / (q - 1)
    return entropy
```