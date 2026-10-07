from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

from corporate_rag.config import ROOT_DIR
from corporate_rag.runtime import create_assistant

DATASET = ROOT_DIR / "data" / "evaluation" / "golden_questions.json"


def load_cases(path: Path = DATASET) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_evaluation_records(cases: list[dict], assistant) -> list[dict]:
    """Run the assistant once per case and retain data for every evaluation stage."""
    records: list[dict] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        print(f"Ответ ассистента: {index}/{total}…", flush=True)
        result = assistant.answer(case["question"])
        records.append(
            {
                "user_input": case["question"],
                "response": result.text,
                "retrieved_contexts": [source.text for source in result.sources],
                "retrieved_sources": [source.metadata.get("source") for source in result.sources],
                "expected_sources": case["expected_sources"],
                "reference": case.get("ground_truth", ""),
            }
        )
    return records


def retrieval_metrics(records: list[dict]) -> dict[str, float]:
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    for record in records:
        sources = record["retrieved_sources"]
        expected = set(record["expected_sources"])
        ranks = [index + 1 for index, source in enumerate(sources) if source in expected]
        recall_hits += int(bool(ranks))
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
    total = len(records) or 1
    return {"Recall@3": recall_hits / total, "MRR@3": sum(reciprocal_ranks) / total}


def run_ragas(records: list[dict]) -> dict:
    import instructor
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import InstructorLLM
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    from corporate_rag.config import settings

    judge_client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )
    ragas_embeddings = HuggingFaceEmbeddings(
        model=settings.embedding_model,
        cache_folder=str(settings.embedding_cache_dir),
        normalize_embeddings=False,
    )
    # The local OpenAI-compatible server accepts structured output only as
    # response_format={"type": "json_schema"}; Instructor's JSON mode sends
    # the unsupported legacy "json_object" format.
    ragas_llm = InstructorLLM(
        client=instructor.from_openai(judge_client, mode=instructor.Mode.JSON_SCHEMA),
        model=os.getenv("JUDGE_MODEL", settings.llm_model),
        provider="openai",
        temperature=0,
        max_tokens=settings.llm_max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    metrics = {
        "faithfulness": (
            Faithfulness(llm=ragas_llm),
            ("user_input", "response", "retrieved_contexts"),
        ),
        "answer_relevancy": (
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ("user_input", "response"),
        ),
        "context_precision_with_reference": (
            ContextPrecisionWithReference(llm=ragas_llm),
            ("user_input", "reference", "retrieved_contexts"),
        ),
        "context_recall": (
            ContextRecall(llm=ragas_llm),
            ("user_input", "reference", "retrieved_contexts"),
        ),
    }

    async def score_metric(name: str, metric, fields: tuple[str, ...]) -> tuple[float, int]:
        eligible = [record for record in records if all(record.get(field) for field in fields)]
        if not eligible:
            print(f"RAGAS {name}: нет подходящих примеров.", flush=True)
            return float("nan"), 0

        total = len(eligible)
        print(f"RAGAS {name}: 0/{total}…", flush=True)
        tasks = [
            metric.ascore(**{field: record[field] for field in fields}) for record in eligible
        ]
        scores = []
        failures = []
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                scores.append(await task)
            except Exception as error:  # noqa: BLE001 - aggregate judge errors by metric
                failures.append(str(error))
            print(f"RAGAS {name}: {completed}/{total}", flush=True)
        if failures:
            raise RuntimeError(f"RAGAS metric '{name}' failed: {'; '.join(failures)}")

        values = [float(score.value) for score in scores if not math.isnan(float(score.value))]
        return (sum(values) / len(values) if values else float("nan")), len(eligible)

    async def score_all() -> dict:
        try:
            results = {
                name: await score_metric(name, metric, fields)
                for name, (metric, fields) in metrics.items()
            }
            return {
                **{name: score for name, (score, _) in results.items()},
                "evaluated_samples": {name: count for name, (_, count) in results.items()},
            }
        finally:
            await judge_client.close()

    return asyncio.run(score_all())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval and optionally RAGAS judge metrics.")
    parser.add_argument("--with-ragas", action="store_true", help="Call the configured LLM-as-a-Judge.")
    args = parser.parse_args()
    print("Загрузка индекса и моделей…", flush=True)
    assistant = create_assistant()
    cases = load_cases()
    records = collect_evaluation_records(cases, assistant)
    report = retrieval_metrics(records)
    if args.with_ragas:
        print("Запуск RAGAS-метрик…", flush=True)
        report["ragas"] = run_ragas(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))
