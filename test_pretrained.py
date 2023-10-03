import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from enformer_pytorch.deep_seq_finetune import HeadAdapterWrapper
from enformer_pytorch.modeling_enformer import Enformer
from enformer_pytorch.data import MaskedGenomeIntervalDataset
from torch import nn
import torch.nn.init as init
from enformer_pytorch.deep_seq import DeepSeq

# Initialize model and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset and dataloader
dataset = MaskedGenomeIntervalDataset(
    bed_file="./data/atac_SRX5437818.filtered.bed",
    fasta_file="./data/genome.fa",
    pileup_dir="./data/cell_lines/LNCAP/SRX5437818.bowtie.sorted.nodup",
    return_augs=False,
    rc_aug=False,
    shift_augs=(-2, 2),
    mask_prob=0.15,
    context_length=16_384,
)
num_workers = 4
dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True
)  # , num_workers=num_workers)


deepseq = DeepSeq.from_hparams(
    dim=1536,
    depth=11,
    heads=8,
    # output_heads=dict(base_prediction=16_384),
    target_length=512,
    return_augs=True,
    rc_aug=True,
    num_downsamples=5,  # 5 downsamples for 16k context length
).to(device)

# Load pretrained weights
enformer = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
deepseq.transformer.load_state_dict(enformer.transformer.state_dict())
deepseq.final_pointwise.load_state_dict(enformer.final_pointwise.state_dict())

# Freeze transformer weights (we can unfreeze after some training)
for param in deepseq.transformer.parameters():
    param.requires_grad = False


model = HeadAdapterWrapper(
    enformer=deepseq,
    num_tracks=4,  # NUMBER OF CELL LINES
    post_transformer_embed=True,  # by default, embeddings are taken from after the final pointwise block w/ conv -> gelu - but if you'd like the embeddings right after the transformer block with a learned layernorm, set this to True
).to(device)

for i, batch in enumerate(dataloader):
    data = batch[0]
    targets = batch[1]
    print("data:", data)
    print("targets:", targets)
