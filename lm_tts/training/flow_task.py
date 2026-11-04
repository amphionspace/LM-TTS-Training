"""Continuous-latent (dots.tts) training data, flow-matching objective and metrics."""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data.batch import val_batches
from ..data.build import verify_tokenizer_compatibility
from ..data.dots import DotsCollator, DotsFeatureDataset, Layout
from ..objectives.flow import flow_loss
from .distributed import batch_health, move

# The flow-matching head: the DiT and the projections that condition it.
FLOW_HEAD = (
    "velocity_field_predictor",
    "hidden_proj",
    "latent_proj",
    "coordinate_proj",
    "xvec_proj",
)


class FlowTask:
    def __init__(self, config, full_config, assembled, rank, backend):
        self.config, self.settings = config, config["train"]
        settings = self.settings
        tokenizer = AutoTokenizer.from_pretrained(
            assembled, local_files_only=True, fix_mistral_regex=False
        )
        self.vocabulary = len(tokenizer)
        layout = Layout(tokenizer)
        self.collate = DotsCollator(layout)
        self.interleave_fraction = settings.get("interleave_fraction", 0.0)

        def dataset(path):
            return DotsFeatureDataset(
                path,
                layout,
                interleave_fraction=self.interleave_fraction,
                verify_integrity=rank == 0,
            )

        self.train_data = dataset(config["data"]["build"])
        if rank == 0:
            self.train_data.validate_budgets(
                settings["max_batch_frames"], settings["max_batch_tokens"]
            )
        verify_tokenizer_compatibility(assembled, self.train_data.manifest["tokenizer"])
        val_build = config["data"].get("val_build")
        self.val_data = dataset(val_build) if val_build else None
        if self.val_data is not None:
            verify_tokenizer_compatibility(assembled, self.val_data.manifest["tokenizer"])

    def configure(self, model):
        if model.llm.model.embed_tokens.num_embeddings != self.vocabulary:
            raise ValueError("LLM embedding rows disagree with the assembled tokenizer")
        # The release trains with x-vector dropout; an experiment may override the rate.
        rate = self.settings.get("xvector_dropout", model.xvec_drop_rate)
        if type(rate) not in (int, float) or not 0 <= rate <= 1:
            raise ValueError("train.xvector_dropout must be in [0, 1]")
        model.xvec_drop_rate = rate
        # SOAR post-trained only the flow-matching head; freezing it keeps that alignment.
        # Gradients still flow through the frozen head into the LLM and patch encoder.
        frozen = self.settings.get("frozen_modules", [])
        if not set(frozen) <= set(FLOW_HEAD):
            raise ValueError(f"train.frozen_modules must be among {FLOW_HEAD}")
        for name in frozen:
            getattr(model, name).requires_grad_(False)

    def describe(self, model):
        return {
            "interleave_fraction": self.interleave_fraction,
            "xvector_dropout": model.xvec_drop_rate,
            "cfg_dropout": model.cfg_droprate,
            "frozen_modules": self.settings.get("frozen_modules", []),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    def begin_step(self, model, batches, device, split):
        self.counts = torch.tensor(
            [
                sum(
                    (b["frame_lengths"] - b["reference_intervals"].diff().squeeze(1)).sum().item()
                    for b in batches
                ),
                sum(len(b["frame_lengths"]) for b in batches),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(self.counts)
        self.sums = torch.zeros(2, dtype=torch.float64, device=device)

    def loss(self, model, batch, world):
        out = flow_loss(model(batch), batch)
        # FM averages over every supervised patch of the update and EOS over every utterance.
        loss = world * (out["fm_sum"] / self.counts[0] + out["eos_sum"] / self.counts[1])
        self.sums += torch.stack([out["fm_sum"], out["eos_sum"]]).detach()
        return loss

    def end_step(self):
        dist.all_reduce(self.sums)

    def metrics(self):
        fm, eos = (self.sums / self.counts).tolist()
        return {
            "fm_loss": fm,
            "eos_loss": eos,
            "objective": fm + eos,
            "patches": self.counts[0].item(),
        }

    @torch.no_grad()
    def validate(self, model, dataset, device):
        """Flow times, noise and posterior samples are drawn from a fixed per-rank seed, so
        successive validations of one run compare the same draws."""
        model.eval()
        torch.manual_seed(self.config["seed"] + dist.get_rank())
        plan = list(
            val_batches(
                dataset, self.config["eval"]["batch_size"], dist.get_world_size(), dist.get_rank()
            )
        )
        loader = DataLoader(
            dataset,
            batch_sampler=[indices for indices, _ in plan],
            collate_fn=self.collate,
            num_workers=0,
        )
        # fm sum, eos sum, patches, utterances, skipped, discarded
        totals = torch.zeros(6, dtype=torch.float64, device=device)
        for (_, real), batch in zip(plan, loader, strict=True):
            batch = move(batch, device)
            if real:
                totals[4] += batch["skipped_samples"]
            if not batch_health(batch, device)[0]:
                if real:
                    totals[5] += len(batch.get("frame_lengths", ()))
                continue
            out = flow_loss(model(batch), batch)
            if real:
                totals[:4] += torch.stack(
                    [out["fm_sum"], out["eos_sum"], out["patch_count"], out["sample_count"]]
                ).double()
        dist.all_reduce(totals)
        if totals[3] == 0:
            raise ValueError("No usable validation samples remain")
        model.train()
        fm, eos = (totals[0] / totals[2]).item(), (totals[1] / totals[3]).item()
        return {
            "fm_loss": fm,
            "eos_loss": eos,
            "loss": fm + eos,
            "skipped_samples": totals[4].item(),
            "discarded_samples": totals[5].item(),
        }
