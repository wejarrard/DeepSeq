import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import polars as pl
import numpy as np
from random import randrange, random
from pathlib import Path
from pyfaidx import Fasta

# helper functions


def exists(val):
    return val is not None


def identity(t):
    return t


def cast_list(t):
    return t if isinstance(t, list) else [t]


def coin_flip():
    return random() > 0.5


# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord("a")] = 0
seq_indices_embed[ord("c")] = 1
seq_indices_embed[ord("g")] = 2
seq_indices_embed[ord("t")] = 3
seq_indices_embed[ord("n")] = 4
seq_indices_embed[ord("A")] = 0
seq_indices_embed[ord("C")] = 1
seq_indices_embed[ord("G")] = 2
seq_indices_embed[ord("T")] = 3
seq_indices_embed[ord("N")] = 4
seq_indices_embed[ord(".")] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()


def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(
        map(lambda t: np.frombuffer(t.encode(), dtype=np.uint8), seq_strs)
    )
    seq_chrs = list(
        map(lambda t: torch.from_numpy(t.copy()), np_seq_chrs)
    )  # Updated to copy the numpy array
    return torch.stack(seq_chrs) if batched else seq_chrs[0]


def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]


def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]


def seq_indices_to_one_hot(t, padding=-1):
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


# augmentations


def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims=(-1,))


def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, "must be one hot encoding with last dimension equal to 4"
    return torch.flip(one_hot, (-1, -2))


# PILEUP PROCESSING
def process_pileups(pileup_dir, chr_name, start, end):
    pileup_file = pileup_dir / f"{chr_name}.pileup"
    assert pileup_file.exists(), f"pileup file for {chr_name} does not exist"

    # Read the file with polars using the correct separator and new_columns arguments
    df = pl.read_csv(
        pileup_file,
        separator="\t",
        has_header=False,
        new_columns=[
            "chr_name",
            "position",
            "nucleotide",
            "count",
            "info",
            "quality",
        ],
    )

    # Filter the DataFrame based on the start and end positions
    df = df.filter((df["position"] >= (start)) & (df["position"] <= (end)))

    return df


class GenomicInterval:
    def __init__(
        self,
        *,
        fasta_file,
        pileup_dir,
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

        self.pileup_dir = Path(pileup_dir)
        assert self.pileup_dir.exists(), "path to pileup directory must exist"

    def __call__(self, chr_name, start, end, return_augs=False):
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = ("." * left_padding) + str(chromosome[start:end]) + ("." * right_padding)

        should_rc_aug = self.rc_aug and coin_flip()

        if self.return_seq_indices:
            seq = str_to_seq_indices(seq)

            if should_rc_aug:
                seq = seq_indices_reverse_complement(seq)

            return seq

        one_hot = str_to_one_hot(seq)

        # Initialize a column of zeros for the reads
        reads_tensor = torch.zeros((end - start, 1), dtype=torch.float)
        extended_data = torch.cat((one_hot, reads_tensor), dim=-1)

        df = process_pileups(self.pileup_dir, chr_name, start, end)

        # Iterate over the rows of the filtered DataFrame and update the reads_tensor with count data
        for row in df.iter_rows(named=True):
            position = row["position"]
            count = row["count"]

            # Calculate the relative position directly without using a separate position_tensor
            relative_position = position - start - 1

            # Update the respective position in the extended_data tensor
            extended_data[relative_position, 4] = count

        if not return_augs:
            return extended_data

        if should_rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])

        return extended_data, rand_shift_tensor, rand_aug_bool_tensor


class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        pileup_dir,
        filter_df_fn=identity,
        chr_bed_to_fasta_map=dict(),
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
    ):
        super().__init__()

        # Initialization for GenomeIntervalDataset
        bed_path = Path(bed_file)
        assert bed_path.exists(), "path to .bed file must exist"

        df = pl.read_csv(str(bed_path), separator="\t", has_header=False)
        df = filter_df_fn(df)
        self.df = df

        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map
        self.return_augs = return_augs

        self.processor = GenomicInterval(
            fasta_file=fasta_file,
            pileup_dir=pileup_dir,
            context_length=context_length,
            return_seq_indices=return_seq_indices,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end = (interval[0], interval[1], interval[2])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)

        return self.processor(chr_name, start, end, return_augs=self.return_augs)

    def __len__(self):
        return len(self.df)


def mask_sequence(input_tensor, mask_prob=0.15, mask_value=-1):
    """
    Masks the input sequence tensor with given probability.
    Only masks the bases and other columns, leaving the last column untouched.
    """
    # Extract the part to mask (all columns except the last one)
    sequence_tensor = input_tensor[:, :-1].clone()
    labels = sequence_tensor.clone()

    # Calculate mask
    mask_rows = torch.bernoulli(
        torch.ones((sequence_tensor.shape[0], 1)) * mask_prob
    ).bool()
    mask = mask_rows.repeat(1, sequence_tensor.shape[1])

    # Replace the original sequence tensor with masked tensor where mask is True
    sequence_tensor[mask] = mask_value
    labels[~mask] = -1  # only calculate loss on masked tokens

    # Construct the masked tensor by concatenating masked sequence and the last column
    masked_tensor = torch.cat([sequence_tensor, input_tensor[:, -1:]], dim=-1)

    return masked_tensor, labels


class MaskedGenomeIntervalDataset(GenomeIntervalDataset):
    def __init__(self, mask_prob=0.15, *args, **kwargs):
        super(MaskedGenomeIntervalDataset, self).__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, index):
        seq, shift, aug = super(MaskedGenomeIntervalDataset, self).__getitem__(index)

        # Mask the sequence and get the labels
        masked_seq, labels = mask_sequence(seq, mask_prob=self.mask_prob)

        return masked_seq, labels, shift, aug


if __name__ == "__main__":
    bed_file = "../data/atac_SRX5437818.filtered.bed"
    fasta_file = "../data/genome.fa"
    pileup_dir = "../data/pileups/SRX5437818.bowtie.sorted.nodup"
    torch.manual_seed(42)

    try:
        dataset = GenomeIntervalDataset(
            bed_file,
            fasta_file,
            pileup_dir,
            return_augs=True,
            rc_aug=True,
            shift_augs=(-2, 2),
        )

        extended_data, shift, aug = dataset[0]

        print("Extended Data:")
        print(extended_data)

        masked_dataset = MaskedGenomeIntervalDataset(
            bed_file=bed_file,
            fasta_file=fasta_file,
            pileup_dir=pileup_dir,
            return_augs=True,
            rc_aug=True,
            shift_augs=(-2, 2),
            mask_prob=0.15,
        )

        masked_data, labels, shift_masked, aug_masked = masked_dataset[0]

        print("\nMasked Data:")
        print(masked_data)

        print("\nLabels Tensor:")
        print(labels)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
