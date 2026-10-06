from __future__ import annotations

import json
import os
from pathlib import Path

from corporate_rag.config import ROOT_DIR
from corporate_rag.runtime import create_assistant

DATASET = ROOT_DIR / "data" / "evaluation" / "golden_questions.json"


def load_cases(path: Path = DATASET) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_metrics(cases: list[dict], assistant) -> dict[str, float]:
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    for case in cases:
        result = assistant.answer(case["question"])
        sources = [source.metadata.get("source") for source in result.sources]
        expected = set(case["expected_sources"])
        ranks = [index + 1 for index, source in enumerate(sources) if source in expected]
        recall_hits += int(bool(ranks))
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
    total = len(cases) or 1
    return {"Recall@3": recall_hits / total, "MRR@3": sum(reciprocal_ranks) / total}


def collect_ragas_records(cases: list[dict], assistant) -> list[dict]:
    records: list[dict] = []
    for case in cases:
        result = assistant.answer(case["question"])
        records.append(
            {
                "user_input": case["question"],
                "response": result.text,
                "retrieved_contexts": [source.text for source in result.sources],
                "reference": case.get("ground_truth", ""),
            }
        )
    return records


def run_ragas(records: list[dict]) -> dict:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    from corporate_rag.config import settings

    judge = ChatOpenAI(
        model=os.getenv("JUDGE_MODEL", settings.llm_model),
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
    )
    embedding = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=str(settings.embedding_cache_dir),
    )
    ragas_llm = LangchainLLMWrapper(judge)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding)
    dataset = EvaluationDataset(samples=[SingleTurnSample(**record) for record in records])
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecisionWithReference(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval and optionally RAGAS judge metrics.")
    parser.add_argument("--with-ragas", action="store_true", help="Call the configured LLM-as-a-Judge.")
    args = parser.parse_args()
    assistant = create_assistant()
    cases = load_cases()
    report = retrieval_metrics(cases, assistant)
    if args.with_ragas:
        report["ragas"] = run_ragas(collect_ragas_records(cases, assistant))
    print(json.dumps(report, ensure_ascii=False, indent=2))
