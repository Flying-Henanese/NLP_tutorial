"""用自回归语言模型计算直接 NLL、PPL、条件 NLL 和 IFD。

直接打分只读候选文本，条件打分另外提供翻译指令和原文。
两次 NLL 都对同一批候选 token 取平均，也支持替换原文做错配对照。
"""

import math
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

TEMPLATE = "Translate the following {source_language} text into {target_language}.\nSource: {source}\nTranslation:\n"


class ProbabilityScorer:
    def __init__(self, model_id, revision, max_tokens, device):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Install the gpu extra and check nvidia-smi.")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        loader = AutoModelForCausalLM
        # Qwen3.5 的完整检查点包含视觉编码器，按检查点的架构加载。
        # 打分时只传纯文本 token，直接读取语言模型的 logits。
        if config.model_type == "qwen3_5" and (
            "Qwen3_5ForConditionalGeneration" in (config.architectures or [])
        ):
            loader = AutoModelForImageTextToText
        self.model = loader.from_pretrained(
            model_id, config=config, revision=revision, dtype=dtype
        ).to(self.device).eval()
        model_limit = getattr(config.get_text_config(), "max_position_embeddings", max_tokens)
        self.max_tokens = min(model_limit, max_tokens)
        self.metadata = {
            "model": model_id,
            "requested_revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "dtype": str(dtype),
            "torch": torch.__version__,
            "max_tokens": self.max_tokens,
            "translation_template": TEMPLATE,
        }

    @torch.inference_mode()
    def measure(self, answer_ids, prefix_ids):
        start = perf_counter()
        if len(answer_ids) < 2:
            raise ValueError("candidate has fewer than two tokens")
        if len(prefix_ids) + len(answer_ids) > self.max_tokens:
            raise ValueError(f"input exceeds {self.max_tokens} tokens")
        on_gpu = self.device.type == "cuda"
        if on_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)
        ids = torch.tensor([prefix_ids + answer_ids], device=self.device)
        logits = self.model(input_ids=ids, use_cache=False).logits[0]
        # Score candidate tokens 1..N-1 in both runs. Row i predicts token i+1.
        offset = len(prefix_ids)
        losses = cross_entropy(logits[offset:-1].float(), ids[0, offset + 1:], reduction="none")
        nll = losses.mean().item()
        if not math.isfinite(nll):
            raise ValueError("nonfinite loss")
        return {
            "nll": nll,
            "seconds": perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if on_gpu else None,
            "first_token_nll": losses[0].item(),
        }

    def conditional(self, sample, answer_ids, source):
        if not source.strip():
            raise ValueError("source is empty")
        prefix = TEMPLATE.format(
            source=source,
            source_language=sample.get("source_language", "Chinese"),
            target_language=sample.get("target_language", "English"),
        )
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        return self.measure(answer_ids, prefix_ids)

    def score(self, sample, alternate_source=None):
        started = perf_counter()
        result = {}
        try:
            text = sample["candidate"]
            if not text.strip():
                raise ValueError("candidate is empty")
            answer_ids = self.tokenizer.encode(text, add_special_tokens=False)
            result["direct"] = self.measure(answer_ids, [])
            direct_nll = result["direct"]["nll"]
            result["ppl"] = math.exp(direct_nll)
            result["token_count"] = len(answer_ids) - 1
            result["token_check"] = {
                "previous_id": answer_ids[0],
                "target_id": answer_ids[1],
                "target_text": self.tokenizer.decode(answer_ids[1:2]),
            }
        except (ValueError, OverflowError, torch.cuda.OutOfMemoryError) as error:
            result["direct_error"] = str(error).splitlines()[0]

        if "ppl" in result:
            for name, source in (("conditional", sample["source"]), ("mismatch", alternate_source)):
                if source is None:
                    continue
                try:
                    result[name] = self.conditional(sample, answer_ids, source)
                    if direct_nll > 1e-8:
                        result[name]["ifd"] = result[name]["nll"] / direct_nll
                    else:
                        result[name]["ifd_error"] = "direct NLL is too close to zero"
                except (ValueError, torch.cuda.OutOfMemoryError) as error:
                    result[name + "_error"] = str(error).splitlines()[0]
        result["seconds"] = perf_counter() - started
        return result
