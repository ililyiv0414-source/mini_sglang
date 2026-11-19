"""Offline correctness reference; this is not a performance benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference.json"))
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                            trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    results = []
    first_logits = []
    for prompt in ["What is the capital of France? Answer briefly.",
                   "Explain the purpose of a KV cache in one sentence."]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            logits = model(**inputs, use_cache=True).logits[:, -1, :].float().cpu()
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                    use_cache=True, pad_token_id=tokenizer.eos_token_id)
        generated = output[0, inputs.input_ids.shape[1]:].tolist()
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Prompt: {prompt}\nAnswer: {answer}\n", flush=True)
        results.append({"prompt": prompt, "formatted_prompt": text,
                        "input_ids": inputs.input_ids[0].tolist(),
                        "output_ids": generated, "answer": answer})
        first_logits.append(logits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logits_path = args.output.with_suffix(".logits.pt")
    torch.save(first_logits, logits_path)
    manifest = args.model / "SHA256SUMS"
    report = {"model_path": str(args.model.resolve()), "torch": torch.__version__,
              "transformers": transformers.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
              "attention": "eager", "do_sample": False, "max_new_tokens": 64,
              "enable_thinking": False, "first_token_logits_file": str(logits_path),
              "model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
              if manifest.exists() else None, "results": results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reference saved: {args.output}")


if __name__ == "__main__":
    main()
