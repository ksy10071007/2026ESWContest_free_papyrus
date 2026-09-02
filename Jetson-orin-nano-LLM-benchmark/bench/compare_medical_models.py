#!/usr/bin/env python3
"""의학용 LLM 다중 모델 비교 스크립트.

이 스크립트는 의학 객관식 벤치마크를 여러 모델에 대해 반복 실행하고,
모델별 정확도와 파싱 실패 내역을 별도 출력 폴더에 저장한다.

실제 추론 연결부는 generate_llm_response()에서 교체하면 된다.
현재 기본 구현은 모델 경로를 기준으로 약간 다른 응답을 흉내 내는 모의 실행 경로를 제공한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# 1) 샘플 의학 벤치마크 데이터셋
# -----------------------------------------------------------------------------
MEDICAL_MCQ_DATA: List[Dict[str, object]] = [
    {
        "question": "68세 남성이 최근 2년간 서서히 진행하는 시야 흐림과 야간 눈부심을 호소한다. 동공 반사가 느려지고 수정체가 뿌옇게 보인다. 가장 가능성이 높은 진단은 무엇인가?",
        "options": {
            "A": "급성 각폐쇄녹내장",
            "B": "황반변성",
            "C": "백내장",
            "D": "시신경염",
        },
        "correct_answer": "C",
    },
    {
        "question": "57세 여성이 점진적인 말초 시야 소실을 호소한다. 안저검사에서 시신경 유두 함몰(cupping)이 관찰되고 안압이 상승해 있다. 가장 적절한 진단은 무엇인가?",
        "options": {
            "A": "망막박리",
            "B": "원발개방각녹내장",
            "C": "포도막염",
            "D": "결막염",
        },
        "correct_answer": "B",
    },
    {
        "question": "당뇨병 병력이 있는 49세 환자의 안저검사에서 미세동맥류, 점상출혈, 경성삼출물이 보인다. 가장 가능성이 높은 질환은 무엇인가?",
        "options": {
            "A": "망막색소변성증",
            "B": "시신경유두부종",
            "C": "당뇨망막병증",
            "D": "각막궤양",
        },
        "correct_answer": "C",
    },
    {
        "question": "환자가 번쩍이는 빛과 비문증을 호소한 뒤, 시야 한쪽에 커튼이 내려오는 느낌을 말한다. 통증은 없다. 가장 가능성이 높은 진단은 무엇인가?",
        "options": {
            "A": "급성 결막염",
            "B": "중심망막정맥폐쇄",
            "C": "유리체출혈",
            "D": "망막박리",
        },
        "correct_answer": "D",
    },
    {
        "question": "시력 저하와 함께 안구 운동 시 통증이 있고, 색각 저하 및 상대구심동공운동장애(RAPD)가 관찰된다. 가장 가능성이 높은 진단은 무엇인가?",
        "options": {
            "A": "시신경염",
            "B": "백내장",
            "C": "노인성 황반변성",
            "D": "전방출혈",
        },
        "correct_answer": "A",
    },
]

SYSTEM_PROMPT = """너는 의학 객관식 문제를 푸는 한국어 의료 추론 모델이다.

반드시 아래 3개 섹션만 순서대로 출력해야 한다.
다른 제목, 설명, 머리말, 꼬리말, 코드블록, 목록은 절대 추가하지 마라.

1. 증상 분석 (Symptom Analysis)
2. 의학적 추론 과정 (Medical Reasoning)
3. 최종 정답 (Final Answer: format must be exactly like "정답: (A)")

최종 정답 줄은 반드시 정확히 `정답: (A)` 형식을 따라야 하며,
A, B, C, D 중 하나만 괄호 안에 넣어라.
"""

DEFAULT_OUTPUT_DIR = Path("outputs") / "medical_benchmark"


# -----------------------------------------------------------------------------
# 2) 데이터 구조
# -----------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """비교 대상 모델 1개를 나타내는 설정."""

    name: str
    path: str
    n_gpu_layers: Optional[int] = None
    n_ctx: Optional[int] = None
    max_tokens: Optional[int] = None


@dataclass
class EvaluationResult:
    """문항별 채점 결과."""

    index: int
    question: str
    correct_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    reasoning_failure: Optional[str]
    raw_response: str


@dataclass
class ModelRunResult:
    """모델별 전체 실행 결과."""

    model_name: str
    model_path: str
    total_questions: int
    correct_answers: int
    accuracy: float
    reasoning_failures: List[Dict[str, object]]
    items: List[Dict[str, object]]


# -----------------------------------------------------------------------------
# 3) 입력 및 파싱
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""

    parser = argparse.ArgumentParser(description="의학 전용 다중 모델 비교 스크립트")
    parser.add_argument("--models-file", required=True, help="CSV 파일 경로: name,path[,n_gpu_layers,n_ctx,max_tokens]")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "compare"), help="결과 저장 폴더")
    parser.add_argument("--output-csv", default=None, help="모델별 요약 CSV 경로")
    parser.add_argument("--details-json", default=None, help="모델별 상세 JSON 경로")
    parser.add_argument("--limit", type=int, default=0, help="데이터셋 일부만 테스트할 때 사용하는 상한값")
    return parser.parse_args()


def to_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def load_model_specs(models_file: str) -> List[ModelSpec]:
    """모델 CSV를 읽어 비교 대상 목록을 만든다."""

    if not os.path.isfile(models_file):
        raise FileNotFoundError(f"Models CSV not found: {models_file}")

    specs: List[ModelSpec] = []
    with open(models_file, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Models CSV is empty.")

        required = {"name", "path"}
        if not required.issubset({name.strip() for name in reader.fieldnames}):
            raise ValueError("Models CSV must include headers: name,path")

        for row in reader:
            name = (row.get("name") or "").strip()
            path = (row.get("path") or "").strip()
            if not name or not path:
                continue
            specs.append(
                ModelSpec(
                    name=name,
                    path=path,
                    n_gpu_layers=to_optional_int(row.get("n_gpu_layers")),
                    n_ctx=to_optional_int(row.get("n_ctx")),
                    max_tokens=to_optional_int(row.get("max_tokens")),
                )
            )

    if not specs:
        raise ValueError(f"No valid model rows found in {models_file}")

    return specs


# -----------------------------------------------------------------------------
# 4) 프롬프트 및 모의 추론
# -----------------------------------------------------------------------------
def build_cot_prompt(item: Dict[str, object]) -> str:
    """의학 CoT 형식 프롬프트를 구성한다."""

    question = str(item["question"])
    options = item["options"]

    option_lines = [f"{choice}. {options[choice]}" for choice in ("A", "B", "C", "D")]
    user_prompt = "\n".join(
        [
            "[문제]",
            question,
            "",
            "[선택지]",
            *option_lines,
            "",
            "[출력 규칙]",
            "1. 증상 분석 (Symptom Analysis)",
            "2. 의학적 추론 과정 (Medical Reasoning)",
            "3. 최종 정답 (Final Answer: format must be exactly like \"정답: (A)\")",
        ]
    )
    return f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}"


def _extract_question_from_prompt(prompt: str) -> str:
    match = re.search(r"\[문제\]\s*(.*?)\s*\n\s*\[선택지\]", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _mock_answer_from_question(question: str, model_name: str) -> str:
    """샘플 데이터에 맞춘 간단한 키워드 기반 모의 정답 생성기."""

    lowered = question.lower()
    model_hint = model_name.lower()

    keyword_answer_pairs: Sequence[Tuple[Sequence[str], str]] = [
        (("백내장", "cataract", "수정체", "뿌옇"), "C"),
        (("녹내장", "glaucoma", "시신경 유두", "함몰", "cup"), "B"),
        (("당뇨망막병증", "diabetic retinopathy", "미세동맥류", "점상출혈"), "C"),
        (("망막박리", "retinal detachment", "커튼", "비문증", "번쩍"), "D"),
        (("시신경염", "optic neuritis", "안구 운동 시 통증", "rapd", "색각"), "A"),
    ]

    for keywords, answer in keyword_answer_pairs:
        if any(keyword in lowered for keyword in keywords):
            if "fail" in model_hint and answer == "B":
                return "C"
            return answer

    return "A"


def generate_llm_response(prompt: str, model_spec: ModelSpec) -> str:
    """모델별 응답 생성 함수의 자리 표시자.

    실제 환경에서는 여기에서 HuggingFace/vLLM/Ollama 호출로 교체하면 된다.
    현재 구현은 모델 이름과 질문 키워드에 따라 일관된 모의 응답을 만든다.
    """

    question = _extract_question_from_prompt(prompt)
    answer = _mock_answer_from_question(question, model_spec.name)

    return (
        "1. 증상 분석 (Symptom Analysis)\n"
        f"- 모델 {model_spec.name} 기준으로 문제의 핵심 증상을 분석한다.\n\n"
        "2. 의학적 추론 과정 (Medical Reasoning)\n"
        "- 증상, 병력, 안저 소견을 종합하여 가장 타당한 진단을 고른다.\n\n"
        "3. 최종 정답 (Final Answer)\n"
        f"정답: ({answer})"
    )


def parse_final_answer(response: str) -> Tuple[Optional[str], Optional[str]]:
    """응답에서 최종 정답을 파싱한다."""

    final_block_match = re.search(r"3\.\s*최종 정답[\s\S]*$", response, flags=re.MULTILINE)
    if not final_block_match:
        return None, "최종 정답 블록을 찾지 못함"

    final_block = final_block_match.group(0)
    answer_match = re.search(r"정답\s*:\s*\(([ABCD])\)", final_block)
    if not answer_match:
        return None, "최종 정답 형식 불일치"

    predicted = answer_match.group(1)
    for pattern in [r"1\.\s*증상 분석", r"2\.\s*의학적 추론 과정", r"3\.\s*최종 정답"]:
        if not re.search(pattern, response):
            return predicted, f"필수 섹션 누락: {pattern}"

    return predicted, None


# -----------------------------------------------------------------------------
# 5) 평가 및 저장
# -----------------------------------------------------------------------------
def evaluate_dataset(dataset: Sequence[Dict[str, object]], model_spec: ModelSpec) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []

    for index, item in enumerate(dataset, start=1):
        prompt = build_cot_prompt(item)
        response = generate_llm_response(prompt, model_spec)
        predicted_answer, failure_reason = parse_final_answer(response)
        correct_answer = str(item["correct_answer"])
        is_correct = predicted_answer == correct_answer

        if predicted_answer is None and failure_reason is None:
            failure_reason = "예측 정답 파싱 실패"

        results.append(
            EvaluationResult(
                index=index,
                question=str(item["question"]),
                correct_answer=correct_answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                reasoning_failure=failure_reason,
                raw_response=response,
            )
        )

    return results


def summarize_results(results: Sequence[EvaluationResult], model_spec: ModelSpec) -> ModelRunResult:
    total_questions = len(results)
    correct_answers = sum(1 for result in results if result.is_correct)
    accuracy = (correct_answers / total_questions * 100.0) if total_questions else 0.0

    reasoning_failures = [
        {
            "index": result.index,
            "predicted_answer": result.predicted_answer,
            "correct_answer": result.correct_answer,
            "reasoning_failure": result.reasoning_failure,
        }
        for result in results
        if result.reasoning_failure is not None or result.predicted_answer is None
    ]

    items = [
        {
            "index": result.index,
            "question": result.question,
            "predicted_answer": result.predicted_answer,
            "correct_answer": result.correct_answer,
            "is_correct": result.is_correct,
            "reasoning_failure": result.reasoning_failure,
            "raw_response": result.raw_response,
        }
        for result in results
    ]

    return ModelRunResult(
        model_name=model_spec.name,
        model_path=model_spec.path,
        total_questions=total_questions,
        correct_answers=correct_answers,
        accuracy=accuracy,
        reasoning_failures=reasoning_failures,
        items=items,
    )


def print_model_summary(summary: ModelRunResult) -> None:
    print(f"\n==================== {summary.model_name} ====================")
    print(f"모델 경로: {summary.model_path}")
    print(f"총 문항 수: {summary.total_questions}")
    print(f"정답 수: {summary.correct_answers}")
    print(f"정확도: {summary.accuracy:.2f}%")

    if summary.reasoning_failures:
        print("파싱/형식 실패 내역: 있음")
        for failure in summary.reasoning_failures:
            print(
                f"- [{failure['index']}] 예측={failure['predicted_answer'] or 'N/A'} / 정답={failure['correct_answer']} / 사유={failure['reasoning_failure']}"
            )
    else:
        print("파싱/형식 실패 내역: 없음")


def save_outputs(summaries: Sequence[ModelRunResult], output_dir: str, output_csv: Optional[str], details_json: Optional[str]) -> None:
    """모델별 요약과 상세 결과를 파일로 저장한다."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_csv) if output_csv else target_dir / "medical_compare_summary.csv"
    json_path = Path(details_json) if details_json else target_dir / "medical_compare_details.json"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_name", "model_path", "total_questions", "correct_answers", "accuracy"],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "model_name": summary.model_name,
                    "model_path": summary.model_path,
                    "total_questions": summary.total_questions,
                    "correct_answers": summary.correct_answers,
                    "accuracy": f"{summary.accuracy:.2f}",
                }
            )

    json_payload = [
        {
            "model_name": summary.model_name,
            "model_path": summary.model_path,
            "total_questions": summary.total_questions,
            "correct_answers": summary.correct_answers,
            "accuracy": round(summary.accuracy, 2),
            "reasoning_failures": summary.reasoning_failures,
            "items": summary.items,
        }
        for summary in summaries
    ]

    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[INFO] 요약 CSV 저장: {csv_path}")
    print(f"[INFO] 상세 JSON 저장: {json_path}")


# -----------------------------------------------------------------------------
# 6) 메인
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    model_specs = load_model_specs(args.models_file)
    dataset = MEDICAL_MCQ_DATA[: args.limit] if args.limit and args.limit > 0 else MEDICAL_MCQ_DATA

    summaries: List[ModelRunResult] = []

    for index, model_spec in enumerate(model_specs, start=1):
        print(f"\n=== [{index}/{len(model_specs)}] 의료 벤치마크 실행: {model_spec.name} ===")
        results = evaluate_dataset(dataset, model_spec)
        summary = summarize_results(results, model_spec)
        summaries.append(summary)
        print_model_summary(summary)

    save_outputs(summaries, args.output_dir, args.output_csv, args.details_json)

    print("\n[DONE] 의료 전용 다중 모델 비교가 완료되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())