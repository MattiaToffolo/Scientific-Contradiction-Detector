"""Evaluate stance classification on reproducible SciFact development subsets."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import time
import urllib.request


LABELS = ("SUPPORT", "CONTRADICT", "NEUTRAL")
DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
BASELINE_PROMPT = (
    "Read the scientific abstract and determine if it SUPPORTS, CONTRADICTS, "
    "or is NEUTRAL toward the claim below.\n\n"
    "Claim: \"{claim}\"\n\nAbstract: \"{abstract}\"\n\n"
    "Reply with exactly one word \u2014 SUPPORT, CONTRADICT, or NEUTRAL \u2014 and nothing else."
)


def compute_metrics(rows):
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        matrix[row["gold"]][row["prediction"]] += 1
    per_class = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in LABELS)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": support,
        }
    total = len(rows)
    accuracy = sum(matrix[label][label] for label in LABELS) / total
    z_score = 1.96
    denominator = 1 + z_score ** 2 / total
    centre = (accuracy + z_score ** 2 / (2 * total)) / denominator
    margin = z_score * (accuracy * (1 - accuracy) / total + z_score ** 2 / (4 * total ** 2)) ** 0.5 / denominator
    return {
        "n": total,
        "accuracy": accuracy,
        "accuracy_wilson_95_interval": [centre - margin, centre + margin],
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABELS),
        "majority_class_accuracy": max(Counter(row["gold"] for row in rows).values()) / total,
        "per_class": per_class,
        "confusion_matrix_gold_rows_prediction_columns": matrix,
    }


def load_sample(output, per_class, seed, exclude_sample=None):
    archive_path = output / "scifact_data.tar.gz"
    if not archive_path.exists():
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            archive_path.write_bytes(response.read())
    archive_bytes = archive_path.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        def read_jsonl(filename):
            member = next(item for item in archive.getmembers() if item.name.endswith("/" + filename) or item.name == filename)
            with archive.extractfile(member) as stream:
                return [json.loads(line) for line in stream if line.strip()]

        corpus = {str(doc["doc_id"]): doc for doc in read_jsonl("corpus.jsonl")}
        claims = read_jsonl("claims_dev.jsonl")
    candidates = {label: [] for label in LABELS}
    excluded = json.loads(exclude_sample.read_text(encoding="utf-8")) if exclude_sample else []
    excluded_claims = {row["claim_id"] for row in excluded}
    excluded_documents = {row["doc_id"] for row in excluded}
    for claim in claims:
        if claim["id"] in excluded_claims:
            continue
        evidence = claim["evidence"]
        for doc_id in sorted(set(map(str, claim["cited_doc_ids"])) | set(evidence)):
            if doc_id in excluded_documents:
                continue
            annotations = evidence.get(doc_id, [])
            labels = {item["label"] for item in annotations}
            if len(labels) > 1:
                raise ValueError("Conflicting gold labels")
            gold = next(iter(labels)) if labels else "NEUTRAL"
            document = corpus[doc_id]
            candidates[gold].append({
                "claim_id": claim["id"], "doc_id": doc_id,
                "claim": claim["claim"], "title": document["title"],
                "text": " ".join(document["abstract"]), "gold": gold,
                "gold_source": "annotated evidence" if labels else "cited document without annotated evidence",
            })
    generator = random.Random(seed)
    selected = []
    used_claims = set()
    for label in LABELS:
        generator.shuffle(candidates[label])
        count = 0
        for candidate in candidates[label]:
            if candidate["claim_id"] in used_claims:
                continue
            selected.append(candidate)
            used_claims.add(candidate["claim_id"])
            count += 1
            if count == per_class:
                break
        if count != per_class:
            raise ValueError(f"Insufficient distinct claims for {label}: {count}")
    generator.shuffle(selected)
    return selected, hashlib.sha256(archive_bytes).hexdigest()


def self_test():
    perfect = [{"gold": label, "prediction": label} for label in LABELS]
    assert compute_metrics(perfect)["accuracy"] == 1.0
    assert compute_metrics(perfect)["macro_f1"] == 1.0
    neutral = [{"gold": label, "prediction": "NEUTRAL"} for label in LABELS]
    metrics = compute_metrics(neutral)
    assert abs(metrics["accuracy"] - 1 / 3) < 1e-9
    assert abs(metrics["macro_f1"] - 1 / 6) < 1e-9
    from contradiction_engine import ContradictionDetector, parse_stance
    assert parse_stance('{"evidence":"SUPPORT appears in a quote", "reasoning":"Opposite effect", "stance":"CONTRADICT"}')["stance"] == "CONTRADICT"
    for invalid in ('{}', 'SUPPORT', '{"stance":"INVALID"}', '{"stance":"SUPPORT"}'):
        try:
            parse_stance(invalid)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Invalid classifier output was accepted")
    class FakeLLM:
        def __init__(self, response):
            self.response = response
            self.prompt = None

        def invoke(self, prompt):
            self.prompt = prompt
            return self.response

    detector = ContradictionDetector.__new__(ContradictionDetector)
    detector.llm = FakeLLM("CONTRADICT")
    assert detector.classify_stance("claim", [{"text": "x" * 900 + "DO_NOT_INCLUDE"}])[0]["stance"] == "CONTRADICT"
    assert "DO_NOT_INCLUDE" not in detector.llm.prompt
    detector.stance_llm = FakeLLM('{"evidence":"last sentence", "reasoning":"Opposite finding", "stance":"CONTRADICT"}')
    result = detector.classify_stance_structured("claim", [{"text": "x" * 900 + " last sentence"}])[0]
    assert result["stance"] == "CONTRADICT" and result["evidence_verified"]
    assert "last sentence" in detector.stance_llm.prompt
    print("Metric self-tests passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--exclude-sample", type=Path)
    parser.add_argument("--input-sample", type=Path)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.per_class < 1:
        parser.error("per-class must be positive")
    from contradiction_engine import ContradictionDetector, _STANCE_PROMPT, _BASELINE_STANCE_PROMPT, STANCE_LLM_OPTIONS
    from config import LLM_MODEL
    from langchain_ollama import OllamaLLM

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists() or (args.output / "predictions.jsonl").exists():
        parser.error("Output already contains a run; choose a new output directory.")
    if args.input_sample:
        rows = json.loads(args.input_sample.read_text(encoding="utf-8"))
        dataset_hash = None
    else:
        rows, dataset_hash = load_sample(args.output, args.per_class, args.seed, args.exclude_sample)
    (args.output / "sample.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        model_metadata = json.load(response)

    class RecordingLLM:
        def __init__(self):
            self.model = (OllamaLLM(model=LLM_MODEL, temperature=0) if args.baseline
                          else OllamaLLM(model=LLM_MODEL, format="json", **STANCE_LLM_OPTIONS))
            self.last_output = None

        def invoke(self, prompt):
            self.last_output = self.model.invoke(prompt)
            return self.last_output

    detector = ContradictionDetector.__new__(ContradictionDetector)
    detector.stance_llm = RecordingLLM()
    detector.llm = detector.stance_llm
    started = time.perf_counter()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, start=1):
            if args.baseline:
                prediction = detector.classify_stance(row["claim"], [{"text": row["text"]}])[0]
            else:
                prediction = detector.classify_stance_structured(row["claim"], [{"text": row["text"]}])[0]
            row["prediction"] = prediction["stance"]
            row["evidence_verified"] = prediction.get("evidence_verified")
            row["reasoning"] = prediction.get("reasoning")
            row["raw_response"] = detector.stance_llm.last_output
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"{index}/{len(rows)} gold={row['gold']} predicted={row['prediction']}", flush=True)
    report = {
        "dataset": "SciFact dev, seeded class-balanced cited-document subset",
        "dataset_url": DATA_URL, "dataset_sha256": dataset_hash,
        "seed": None if args.input_sample else args.seed, "model": LLM_MODEL, "ollama_models": model_metadata,
        "sample_sha256": hashlib.sha256(json.dumps([
            {key: row[key] for key in ("claim_id", "doc_id", "claim", "text", "gold")}
            for row in rows
        ], sort_keys=True).encode()).hexdigest(),
        "variant": "original baseline" if args.baseline else "structured full abstract",
        "prompt": _BASELINE_STANCE_PROMPT if args.baseline else _STANCE_PROMPT,
        "abstract_character_limit": 900 if args.baseline else None,
        "llm_options": {"temperature": 0} if args.baseline else STANCE_LLM_OPTIONS,
        "input_sample": str(args.input_sample) if args.input_sample else None,
        "excluded_sample": str(args.exclude_sample) if args.exclude_sample else None,
        "classifier_source_sha256": hashlib.sha256(Path(__file__).with_name("contradiction_engine.py").read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Stance-only evaluation: retrieval and report quality are not evaluated.",
            "NEUTRAL means cited documents without annotated evidence, not random negatives.",
            "Small balanced dev subset; not the official SciFact leaderboard protocol.",
            "Public benchmark exposure during LLM pretraining cannot be excluded.",
            "Prompt developed on a separate development sample; no comparison against an NLI baseline.",
        ],
        **compute_metrics(rows),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()