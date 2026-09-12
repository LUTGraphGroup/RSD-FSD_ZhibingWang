import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are a clinical terminologist with expertise in ICD coding and clinical note interpretation.
Return only the requested XML structure.
Do not add markdown fences, explanations, bullets, or extra headings.
Keep each title exactly as provided in the input.
Write in formal clinical language.
"""

PROMPT_TEMPLATE = """For each ICD code title below, generate two paragraphs:

1. A clinical description paragraph:
- explain the code meaning and clinical context
- summarize the main indications, disorders, or pathological processes
- include procedural details when the title refers to a procedure
- when appropriate, naturally include common clinical-note aliases, abbreviations, or expressions that are plausible and not ambiguous

2. A comorbidity-oriented paragraph:
- summarize the top 1-3 major complications or related conditions
- explain their likely causes
- describe the pathophysiological links between the code and those conditions

Return the result for EACH code using exactly the following structure:

<entry>
<title>The input code title</title>
<clinical_description>One paragraph only.</clinical_description>
<comorbidity_profile>One paragraph only.</comorbidity_profile>
</entry>

Rules:
- Keep the title text exactly as provided in the input.
- Do not omit any tag.
- Do not merge multiple codes into one entry.

=====

Input code titles:
{code_titles}
"""


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:xml)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.replace("\r\n", "\n").strip()


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_code_titles(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    items: List[Tuple[str, str]] = []
    if isinstance(data, dict):
        for code, value in data.items():
            title = value[0] if isinstance(value, list) and value else value
            code = normalize_text(code)
            title = normalize_text(title)
            if code and title:
                items.append((code, title))
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            code = normalize_text(row.get("code", ""))
            title = normalize_text(row.get("title", ""))
            if code and title:
                items.append((code, title))
    else:
        raise TypeError("Input must be a JSON dict or a list of objects with code/title fields.")

    return items


def parse_batch_response(response_text: str, original_titles: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    cleaned = normalize_response(response_text)
    blocks = re.findall(r"<entry>\s*(.*?)\s*</entry>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    result_map: Dict[str, Tuple[str, str]] = {}

    for block in blocks:
        title = re.search(r"<title>\s*(.*?)\s*</title>", block, flags=re.DOTALL | re.IGNORECASE)
        desc = re.search(
            r"<clinical_description>\s*(.*?)\s*</clinical_description>",
            block,
            flags=re.DOTALL | re.IGNORECASE,
        )
        comorb = re.search(
            r"<comorbidity_profile>\s*(.*?)\s*</comorbidity_profile>",
            block,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if title and desc and comorb:
            result_map[normalize_title(title.group(1))] = (
                normalize_text(desc.group(1)),
                normalize_text(comorb.group(1)),
            )

    return [result_map.get(normalize_title(title), (None, None)) for title in original_titles]


def generate_batch(
    client: Any,
    model: str,
    code_titles: List[str],
    temperature: float,
    max_tokens: int,
) -> List[Tuple[Optional[str], Optional[str]]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(code_titles="\n".join(code_titles))},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return parse_batch_response(content, code_titles)


def save_checkpoint(path: Path, results: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Checkpoint must be a JSON object: {path}")
    return data


def build_anchor_text(code: str, title: str, clinical_description: str, comorbidity_profile: str) -> str:
    return normalize_text(
        f"ICD code {code}. Label title: {title}. {clinical_description} {comorbidity_profile}"
    )


def run(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the project requirements before generating enhanced text.") from exc

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set {args.api_key_env} before running this script.")

    client = OpenAI(api_key=api_key)
    items = load_code_titles(args.input)
    results = load_checkpoint(args.checkpoint)

    pending = [(code, title) for code, title in items if code not in results]
    print(f"labels: {len(items)}; pending: {len(pending)}")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        titles = [title for _, title in batch]
        generated = generate_batch(
            client=client,
            model=args.model,
            code_titles=titles,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        for (code, title), (clinical_description, comorbidity_profile) in zip(batch, generated):
            clinical_description = clinical_description or ""
            comorbidity_profile = comorbidity_profile or ""
            results[code] = {
                "code": code,
                "title": title,
                "clinical_description": clinical_description,
                "comorbidity_profile": comorbidity_profile,
                "anchor_text": build_anchor_text(code, title, clinical_description, comorbidity_profile),
                "source": f"fixed_prompt_{args.model}",
            }

        save_checkpoint(args.checkpoint, results)
        completed = min(start + len(batch), len(pending))
        print(f"generated: {completed}/{len(pending)}")
        time.sleep(args.sleep)

    ordered = [results[code] for code, _ in items if code in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)
    print(f"saved: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RSD-FSD enhanced ICD label text with a fixed prompt.")
    parser.add_argument("--input", type=Path, required=True, help="JSON dict {code: title/list} or list with code/title.")
    parser.add_argument("--output", type=Path, required=True, help="Enhanced label JSON for RSD-FSD.")
    parser.add_argument("--checkpoint", type=Path, default=Path("enhanced_text_checkpoint.json"))
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--sleep", type=float, default=1.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

