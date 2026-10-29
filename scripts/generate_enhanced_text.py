import argparse
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROMPT_VERSION = "manuscript_context_v2_20261028"

SYSTEM_PROMPT = """You are a clinical terminology expert specializing in International Classification of Diseases (ICD) coding, medical ontology interpretation, and clinical documentation analysis.

Your task is to generate clinically accurate and semantically enriched ICD label descriptions based on official ICD definitions, hierarchical category information, and relevant clinical knowledge.

Follow these principles:

1. **Official ICD Consistency:** Treat the provided official ICD descriptions and coding definitions as the primary authoritative references. Preserve the original diagnostic or procedural meaning of each code.
2. **Hierarchical Consistency:** Respect the provided ICD category hierarchy, including parent categories, subcategories, and diagnostic specificity. Do not infer unsupported hierarchical relationships.
3. **Clinical Scope Preservation:** Do not broaden or narrow the diagnostic scope of an ICD code. Preserve clinical qualifiers, anatomical locations, disease subtypes, severity distinctions, and inclusion or exclusion conditions when specified.
4. **Semantic Enrichment:** Expand the clinical meaning of each label with relevant medical terminology, clinical manifestations, commonly used synonyms, and appropriate pathophysiological background. Avoid unsupported or ambiguous information.
5. **Clinical Reliability:** Do not introduce unrelated diseases, unsupported diagnostic assumptions, fabricated clinical relationships, or misleading medical concepts.
6. **Output Consistency:** Return only the specified XML structure. Preserve each input ICD title exactly. Do not include markdown formatting, explanations, or additional headings.

Use precise, formal, and clinically appropriate language throughout."""

PROMPT_TEMPLATE = """For each ICD code provided below, generate an enhanced semantic description using its official ICD description, hierarchical category information, supplementary semantic information, and clinically relevant pathophysiological background.

The generated description must remain consistent with the official ICD definition and its diagnostic or procedural scope.

For each ICD code, generate two paragraphs:

**1. Clinical Description**

- Explain the official meaning and clinical context of the ICD code.
- Preserve the diagnostic or procedural scope specified by the official description.
- Incorporate relevant hierarchical category information to clarify the code's clinical specificity.
- Summarize the principal diseases, clinical manifestations, pathological processes, or procedural indications associated with the code.
- Include clinically appropriate synonyms, abbreviations, and common expressions used in medical records when they are unambiguous.
- Provide supplementary semantic information that improves the correspondence between ICD terminology and clinical narrative expressions.
- Include relevant procedural details when the code represents a medical procedure.

**2. Comorbidity and Pathophysiological Profile**

- Summarize up to three clinically relevant complications, comorbidities, or associated conditions when supported by established medical knowledge.
- Explain the potential clinical or pathophysiological relationships between the ICD code and these conditions.
- Distinguish associated conditions from the defining diagnostic criteria of the ICD code.
- Do not interpret possible complications or related conditions as confirmed diagnoses.
- Avoid introducing unrelated clinical concepts or unsupported causal relationships.

**3. Clinical and Hierarchical Constraints**

- Strictly preserve the official meaning of each ICD code.
- Do not expand or narrow the diagnostic scope beyond the official ICD definition.
- Ensure consistency with the provided ICD hierarchy and coding version.
- Preserve all explicitly stated diagnostic qualifiers and exclusion conditions.
- Do not introduce unsupported clinical diagnoses, complications, or procedural indications.
- When supplementary information conflicts with the official ICD description, prioritize the official description.
- If specific clinical information is unavailable or uncertain, omit it rather than generating unsupported details.

**4. Required Output Format**

For each code, return exactly the following XML structure:

<entry>
<title>The original ICD code title</title>
<clinical_description>One paragraph containing the clinically enriched description.</clinical_description>
<comorbidity_profile>One paragraph describing relevant comorbidities and pathophysiological relationships.</comorbidity_profile>
</entry>

**5. Output Rules**

- Preserve each original ICD code title exactly as provided.
- Do not omit any required XML tag.
- Generate one independent entry for each ICD code.
- Do not merge multiple ICD codes into one entry.
- Do not include explanations or text outside the XML structure.

=====

**Input ICD Code Information:**

{icd_entries}
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


CONTEXT_FIELDS = (
    "official_description", "hierarchical_category_information",
    "supplementary_semantic_information", "pathophysiological_background",
    "coding_system", "coding_version", "official_description_source",
    "hierarchy_source", "coding_version_source",
)


def load_icd_entries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        rows = []
        for code, value in data.items():
            if isinstance(value, dict):
                if "code" in value and str(value["code"]) != str(code):
                    raise ValueError(f"Conflicting code for {code}")
                rows.append(dict(value, code=str(code)))
            else:
                title = value[0] if isinstance(value, list) and value else value
                rows.append({"code": str(code), "title": title})
    elif isinstance(data, list):
        rows = data
    else:
        raise TypeError("Input must be an object or a list of code/title/context objects")
    items, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each input record must be an object")
        code, title = normalize_text(row.get("code", "")), row.get("title")
        if not code or not isinstance(title, str) or not title.strip() or code in seen:
            raise ValueError("Each record needs a unique code and a nonempty title string")
        seen.add(code)
        entry = {"code": code, "title": title}
        entry.update({field: row.get(field) or None for field in CONTEXT_FIELDS})
        if entry["official_description"] is None:
            entry["official_description"] = title
            entry["official_description_source"] = "input_title_only; authoritative source not supplied"
        items.append(entry)
    if not items:
        raise ValueError("Input contains no ICD entries")
    return items


def load_code_titles(path: Path) -> List[Tuple[str, str]]:
    """Compatibility helper; generation uses full load_icd_entries records."""
    return [(r["code"], r["title"]) for r in load_icd_entries(path)]


def build_messages(entries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                icd_entries=json.dumps(entries, ensure_ascii=False, indent=2))}]


def batches_with_unique_titles(entries, batch_size):
    batch, titles = [], set()
    for entry in entries:
        if len(batch) == batch_size or entry["title"] in titles:
            yield batch
            batch, titles = [], set()
        batch.append(entry)
        titles.add(entry["title"])
    if batch:
        yield batch


def parse_batch_response(response_text: str, original_titles: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    if len(set(original_titles)) != len(original_titles):
        raise ValueError("Duplicate titles must be placed in separate requests")
    try:
        root = ET.fromstring("<root>" + response_text.strip() + "</root>")
    except ET.ParseError as exc:
        raise ValueError("Response is not the required XML structure") from exc
    if root.text and root.text.strip():
        raise ValueError("Text outside entry tags is not allowed")
    result_map: Dict[str, Tuple[str, str]] = {}
    for entry in root:
        tags = [child.tag for child in entry]
        if entry.tag != "entry" or tags != ["title", "clinical_description", "comorbidity_profile"]:
            raise ValueError("Unexpected, duplicate or missing XML tags")
        if (entry.text and entry.text.strip()) or (entry.tail and entry.tail.strip()):
            raise ValueError("Text outside the required XML fields is not allowed")
        if any(list(child) or (child.tail and child.tail.strip()) for child in entry):
            raise ValueError("Nested tags or extra text are not allowed in XML fields")
        title, desc, comorb = (child.text or "" for child in entry)
        if title not in original_titles or title in result_map:
            raise ValueError("Response title changed, duplicated or not requested")
        if not desc.strip() or not comorb.strip():
            raise ValueError("Both requested paragraphs must be nonempty")
        result_map[title] = (normalize_text(desc), normalize_text(comorb))
    if set(result_map) != set(original_titles):
        raise ValueError("Response omitted at least one requested code")
    return [result_map[title] for title in original_titles]


def generate_batch(
    client: Any,
    model: str,
    icd_entries: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    log_path: Optional[Path] = None,
) -> List[Tuple[Optional[str], Optional[str]]]:
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(icd_entries),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if log_path is not None:
        save_checkpoint(log_path, response.model_dump(mode="json"))
    if response.choices[0].finish_reason != "stop":
        raise ValueError("Generation did not finish normally; inspect raw response before retrying")
    content = response.choices[0].message.content or ""
    return parse_batch_response(content, [r["title"] for r in icd_entries])


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
    if args.batch_size < 1 or args.max_tokens < 1 or args.sleep < 0:
        raise ValueError("batch-size/max-tokens must be positive and sleep nonnegative")
    items = load_icd_entries(args.input)
    if args.require_complete_context:
        for entry in items:
            for field in ("official_description", "official_description_source",
                          "hierarchical_category_information", "hierarchy_source",
                          "coding_system", "coding_version", "coding_version_source"):
                if not entry[field] or str(entry[field]).startswith("input_title_only"):
                    raise ValueError(f"{entry['code']}: required sourced context missing: {field}")
    config = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT, "user_template": PROMPT_TEMPLATE,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model_requested": args.model, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "batch_size": args.batch_size,
        "top_p_explicitly_passed": False, "seed_explicitly_passed": False,
    }
    if args.dry_run:
        target = args.output.with_suffix(".dry_run.json")
        save_checkpoint(target, {"state": "DRY_RUN_NO_API_CALL", "configuration": config,
                                "batches": [build_messages(batch) for batch in batches_with_unique_titles(items, args.batch_size)]})
        print(f"saved dry run (no API call): {target}")
        return
    results = load_checkpoint(args.checkpoint)
    checkpoint_manifest = args.checkpoint.with_name(args.checkpoint.stem + ".manifest.json")
    if results:
        if not checkpoint_manifest.exists() or load_checkpoint(checkpoint_manifest) != config:
            raise ValueError("Checkpoint prompt/input/settings differ or lack provenance. Use a new checkpoint path for this template.")
        if not set(results).issubset({entry['code'] for entry in items}):
            raise ValueError("Checkpoint contains codes outside this input")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the project requirements before generating enhanced text.") from exc

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set {args.api_key_env} before running this script.")

    client = OpenAI(api_key=api_key)
    save_checkpoint(checkpoint_manifest, config)
    pending = [entry for entry in items if entry["code"] not in results]
    print(f"labels: {len(items)}; pending: {len(pending)}")
    log_dir = args.log_dir or args.output.parent / (args.output.stem + "_generation_logs")
    completed = 0
    for batch in batches_with_unique_titles(pending, args.batch_size):
        timestamp = datetime.now(timezone.utc)
        batch_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        request_path = log_dir / f"{batch_id}_request.json"
        response_path = log_dir / f"{batch_id}_response.json"
        save_checkpoint(request_path, {"requested_at_utc": timestamp.isoformat(),
                                      "configuration": config,
                                      "codes": [entry["code"] for entry in batch],
                                      "messages": build_messages(batch)})
        generated = generate_batch(
            client=client,
            model=args.model,
            icd_entries=batch,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            log_path=response_path,
        )

        response_meta = load_checkpoint(response_path)
        for entry, (clinical_description, comorbidity_profile) in zip(batch, generated):
            code, title = entry["code"], entry["title"]
            results[code] = {
                "code": code,
                "title": title,
                "clinical_description": clinical_description,
                "comorbidity_profile": comorbidity_profile,
                "anchor_text": build_anchor_text(code, title, clinical_description, comorbidity_profile),
                "source": f"fixed_prompt_{PROMPT_VERSION}_{args.model}",
                "input_context": entry,
                "generation": {"requested_at_utc": timestamp.isoformat(),
                               "model_requested": args.model,
                               "model_returned": response_meta.get("model"),
                               "response_id": response_meta.get("id"),
                               "response_created": response_meta.get("created"),
                               "prompt_version": PROMPT_VERSION,
                               "request_log": str(request_path), "response_log": str(response_path)},
            }

        save_checkpoint(args.checkpoint, results)
        completed += len(batch)
        print(f"generated: {completed}/{len(pending)}")
        time.sleep(args.sleep)

    ordered = [results[entry["code"]] for entry in items]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)
    print(f"saved: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RSD-FSD enhanced ICD label text with a fixed prompt.")
    parser.add_argument("--input", type=Path, required=True, help="JSON code/title/context records; legacy title-only dictionaries are also accepted.")
    parser.add_argument("--output", type=Path, required=True, help="Enhanced label JSON for RSD-FSD.")
    parser.add_argument("--checkpoint", type=Path, default=Path("enhanced_text_checkpoint.json"))
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--log-dir", type=Path, help="Directory for full timestamped request and raw response JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Save filled prompts without importing the SDK or calling the API.")
    parser.add_argument("--require-complete-context", action="store_true", help="Require sourced official description, hierarchy and coding version on every entry.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

