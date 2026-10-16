"""B4: per-user LoRA on raw transcripts — the naive "memory in the weights" baseline.

At each sleep, the loaded user's adapter is (re)trained with next-token loss on everything that
user has said outside incognito sessions; probes are answered by the frozen base plus that adapter,
with no retrieval. This is deliberately the simple recipe: research/landscape.md §8 predicts it
recalls poorly, which is exactly what Engram's sleep recipe has to beat.

Runs on CPU with a tiny model for tests, and on one GPU for real runs. It is single-threaded by
design (one model, adapters swapped per user), so it does not offer ``spawn``: use workers=1.
Needs the ``ml`` extra.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from engram.systems.base import SYSTEM_PROMPT


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 8
    max_len: int = 256
    target_modules: str | list[str] = "all-linear"
    mode: str = "cumulative"  # retrain from scratch on all messages; or "incremental"
    max_new_tokens: int = 96
    seed: int = 0


class LoRAMemory:
    def __init__(
        self, base_model: str, config: LoRAConfig | None = None, device: str | None = None
    ):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model = base_model
        self.cfg = config or LoRAConfig()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(self.device)
        self._lora = LoraConfig(
            r=self.cfg.rank,
            lora_alpha=self.cfg.alpha,
            lora_dropout=self.cfg.dropout,
            target_modules=self.cfg.target_modules,
            task_type="CAUSAL_LM",
        )
        # A permanent, never-trained placeholder so user adapters can be added and deleted freely.
        self.model = get_peft_model(base, self._lora, adapter_name="_none")
        self.model.eval()
        self.adapters: set[str] = set()
        self.logs: dict[str, list[str]] = defaultdict(list)
        self.trained_upto: dict[str, int] = defaultdict(int)
        self.user: str | None = None
        self.context: list[dict] = []
        self.write = True
        self.train_steps = 0

    @property
    def name(self) -> str:
        return f"b4-lora:{self.base_model}"

    @staticmethod
    def _adapter(user_id: str) -> str:
        return "u_" + "".join(c if c.isalnum() else "_" for c in user_id)

    # -- protocol ----------------------------------------------------------------------------

    def load_user(self, user_id: str) -> None:
        self.user = user_id

    def unload_user(self) -> None:
        self.user = None

    def start_session(self, *, write: bool = True, day: int | None = None) -> None:
        self.context, self.write = [], write

    def end_session(self) -> None:
        self.context = []

    def observe(self, message: str) -> None:
        self.context.append({"role": "user", "content": message})
        if self.write and self.user is not None:
            self.logs[self.user].append(message)

    def respond(self, message: str) -> str:
        assert self.user is not None
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.context]
        messages.append({"role": "user", "content": message})
        reply = self._generate(messages)
        self.observe(message)
        self.context.append({"role": "assistant", "content": reply})
        return reply

    def sleep(self) -> None:
        if self.user is None:
            return
        n = len(self.logs[self.user])
        if n > self.trained_upto[self.user]:
            self._train(self.user)
            self.trained_upto[self.user] = n

    def delete_user(self, user_id: str) -> None:
        name = self._adapter(user_id)
        if name in self.adapters:
            self.model.delete_adapter(name)
            self.adapters.discard(name)
        self.logs.pop(user_id, None)
        self.trained_upto.pop(user_id, None)

    def describe(self) -> dict:
        return {"system": "B4", "base_model": self.base_model, **self.cfg.__dict__}

    # -- model -------------------------------------------------------------------------------

    def _format(self, messages: list[dict], *, generation: bool) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False
            )
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        return "\n".join(lines) + ("\nAssistant:" if generation else "")

    def _generate(self, messages: list[dict]) -> str:
        torch = self.torch
        text = self._format(messages, generation=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        name = self._adapter(self.user or "")
        with torch.no_grad():
            if name in self.adapters:
                self.model.set_adapter(name)
                out = self.model.generate(
                    **enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )  # fmt: skip
            else:
                with self.model.disable_adapter():
                    out = self.model.generate(
                        **enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                        pad_token_id=self.tok.pad_token_id,
                    )  # fmt: skip
        return self.tok.decode(
            out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def _train(self, user_id: str) -> None:
        torch = self.torch
        name = self._adapter(user_id)
        cumulative = self.cfg.mode == "cumulative"
        if name in self.adapters and cumulative:
            self.model.delete_adapter(name)
            self.adapters.discard(name)
        if name not in self.adapters:
            self.model.add_adapter(name, self._lora)
            self.adapters.add(name)
        self.model.set_adapter(name)
        marker = f".{name}."
        params = []
        for pname, p in self.model.named_parameters():
            p.requires_grad = marker in pname
            if p.requires_grad:
                params.append(p)

        msgs = (
            self.logs[user_id] if cumulative else self.logs[user_id][self.trained_upto[user_id] :]
        )
        # Raw transcripts: each user message as a one-turn chat, loss on every token.
        texts = [self._format([{"role": "user", "content": m}], generation=False) for m in msgs]
        rng = random.Random(self.cfg.seed)
        opt = torch.optim.AdamW(params, lr=self.cfg.lr)
        self.model.train()
        for _ in range(self.cfg.epochs):
            rng.shuffle(texts)
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = self.tok(
                    texts[i : i + self.cfg.batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=self.cfg.max_len,
                ).to(self.device)  # fmt: skip
                labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
                loss = self.model(**batch, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                self.train_steps += 1
        self.model.eval()
        for p in params:
            p.requires_grad = False

    # -- membership inference (engram.bench.membership) -------------------------------------

    def sequence_losses(
        self, user_id: str, texts: list[str], *, batch_size: int | None = None
    ) -> list[float]:
        """Mean next-token loss of each user message, formatted exactly as ``_train`` formats it,
        with ``user_id``'s adapter active. Raises KeyError if that user has no adapter."""
        name = self._adapter(user_id)
        if name not in self.adapters:
            raise KeyError(f"no adapter for {user_id!r}")
        self.model.set_adapter(name)
        return self._losses(texts, batch_size)

    def base_losses(self, texts: list[str], *, batch_size: int | None = None) -> list[float]:
        """As ``sequence_losses`` but on the frozen base: the reference model in LoRA-Leak."""
        with self.model.disable_adapter():
            return self._losses(texts, batch_size)

    def _losses(self, texts: list[str], batch_size: int | None) -> list[float]:
        torch = self.torch
        bs = batch_size or self.cfg.batch_size
        formatted = [
            self._format([{"role": "user", "content": t}], generation=False) for t in texts
        ]
        out: list[float] = []
        side, self.tok.padding_side = self.tok.padding_side, "right"  # causal: pads never attended
        self.model.eval()
        try:
            with torch.no_grad():
                for i in range(0, len(formatted), bs):
                    batch = self.tok(
                        formatted[i : i + bs], return_tensors="pt", padding=True, truncation=True,
                        max_length=self.cfg.max_len,
                    ).to(self.device)  # fmt: skip
                    logits = self.model(**batch).logits[:, :-1].float()
                    target = batch["input_ids"][:, 1:]
                    mask = batch["attention_mask"][:, 1:].float()
                    nll = torch.nn.functional.cross_entropy(
                        logits.transpose(1, 2), target, reduction="none"
                    )
                    per_text = (nll * mask).sum(1) / mask.sum(1).clamp(min=1)
                    out.extend(per_text.tolist())
        finally:
            self.tok.padding_side = side
        return out
