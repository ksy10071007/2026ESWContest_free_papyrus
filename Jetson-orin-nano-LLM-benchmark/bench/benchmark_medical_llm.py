#!/usr/bin/env python3
"""의학용 CoT 벤치마크용 LLM 평가 스크립트.

MedQA/MMLU 스타일의 객관식 문제를 대상으로,
LLM이 반드시 지정된 한국어 CoT 형식으로 답변하도록 유도한 뒤,
최종 정답을 자동 파싱하여 정확도를 계산한다.

실제 추론 엔진(HuggingFace, vLLM, Ollama 등)은
generate_llm_response() 내부만 교체하면 연결할 수 있도록 설계했다.
"""

from __future__ import annotations

import re
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# 1) 벤치마크용 샘플 데이터셋
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

DEFAULT_OUTPUT_DIR = Path("outputs") / "medical_benchmark"


# -----------------------------------------------------------------------------
# 2) CoT 프롬프트 템플릿
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """너는 의학 객관식 문제를 푸는 한국어 의료 추론 모델이다.

반드시 아래 3개 섹션만 순서대로 출력해야 한다.
다른 제목, 설명, 머리말, 꼬리말, 코드블록, 목록은 절대 추가하지 마라.

1. 증상 분석 (Symptom Analysis)
2. 의학적 추론 과정 (Medical Reasoning)
3. 최종 정답 (Final Answer: format must be exactly like "정답: (A)")

최종 정답 줄은 반드시 정확히 `정답: (A)` 형식을 따라야 하며,
A, B, C, D 중 하나만 괄호 안에 넣어라.
"""


def build_cot_prompt(item: Dict[str, object]) -> str:
    """문제와 선택지를 CoT 형식의 단일 프롬프트 문자열로 변환한다."""

    question = str(item["question"])
    options = item["options"]

    option_lines = []
    for key in ("A", "B", "C", "D"):
        option_lines.append(f"{key}. {options[key]}")

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


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""

    parser = argparse.ArgumentParser(description="의료용 CoT 벤치마크 스크립트")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="결과 파일을 저장할 출력 폴더 경로",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 3) 추론 함수(플러그인 교체 가능)
# -----------------------------------------------------------------------------
def _extract_question_from_prompt(prompt: str) -> str:
    """모의 응답 생성을 위해 프롬프트에서 문제 본문만 추출한다."""

    match = re.search(r"\[문제\]\s*(.*?)\s*\n\s*\[선택지\]", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def _mock_answer_from_question(question: str) -> str:
    """샘플 데이터에 맞춘 간단한 키워드 기반 모의 정답 생성기."""

    lowered = question.lower()

    keyword_answer_pairs: Sequence[Tuple[Sequence[str], str]] = [
        (("백내장", "cataract", "수정체", "뿌옇"), "C"),
        (("녹내장", "glaucoma", "시신경 유두", "함몰", "cup"), "B"),
        (("당뇨망막병증", "diabetic retinopathy", "미세동맥류", "점상출혈"), "C"),
        (("망막박리", "retinal detachment", "커튼", "비문증", "번쩍"), "D"),
        (("시신경염", "optic neuritis", "안구 운동 시 통증", "rapd", "색각"), "A"),
    ]

    for keywords, answer in keyword_answer_pairs:
        if any(keyword in lowered for keyword in keywords):
            return answer

    return "A"


def generate_llm_response(prompt: str) -> str:
    """LLM 추론 함수의 자리 표시자.

    실제 연동 시 이 함수 내부를 HuggingFace/vLLM/Ollama 호출로 교체하면 된다.
    현재 구현은 스크립트가 바로 실행되도록 샘플 문제에 대한 모의 응답을 반환한다.
    """

    question = _extract_question_from_prompt(prompt)
    answer = _mock_answer_from_question(question)

    return (
        "1. 증상 분석 (Symptom Analysis)\n"
        f"- 문제의 핵심 증상은 '{question[:80]}...'처럼 보이며, 임상적 단서를 중심으로 판단한다.\n\n"
        "2. 의학적 추론 과정 (Medical Reasoning)\n"
        "- 제시된 증상과 병력, 안저 소견을 종합하면 가장 잘 맞는 감별진단을 선택해야 한다.\n"
        "- 주어진 선택지 중 임상 양상에 가장 부합하는 진단을 고른다.\n\n"
        "3. 최종 정답 (Final Answer)\n"
        f"정답: ({answer})"
    )


# -----------------------------------------------------------------------------
# 4) 정답 파서 및 평가기
# -----------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """문항별 평가 결과를 저장하는 구조체."""

    index: int
    question: str
    correct_answer: str
    predicted_answer: Optional[str]
    is_correct: bool
    reasoning_failure: Optional[str]
    raw_response: str


def parse_final_answer(response: str) -> Tuple[Optional[str], Optional[str]]:
    """응답에서 최종 정답을 파싱하고, 형식 오류가 있으면 함께 반환한다.

    반환값:
        (예측 정답, 실패 사유)
    """

    final_block_match = re.search(
        r"3\.\s*최종 정답[\s\S]*$",
        response,
        flags=re.MULTILINE,
    )
    if not final_block_match:
        return None, "최종 정답 블록을 찾지 못함"

    final_block = final_block_match.group(0)

    answer_match = re.search(r"정답\s*:\s*\(([ABCD])\)", final_block)
    if not answer_match:
        return None, "최종 정답 형식 불일치"

    predicted = answer_match.group(1)

    required_sections = [
        r"1\.\s*증상 분석",
        r"2\.\s*의학적 추론 과정",
        r"3\.\s*최종 정답",
    ]
    for section_pattern in required_sections:
        if not re.search(section_pattern, response):
            return predicted, f"필수 섹션 누락: {section_pattern}"

    return predicted, None


def evaluate_dataset(dataset: Sequence[Dict[str, object]]) -> List[EvaluationResult]:
    """데이터셋 전체를 순회하며 추론, 파싱, 채점을 수행한다."""

    results: List[EvaluationResult] = []

    for index, item in enumerate(dataset, start=1):
        prompt = build_cot_prompt(item)
        response = generate_llm_response(prompt)
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


# -----------------------------------------------------------------------------
# 5) 리포트 출력
# -----------------------------------------------------------------------------
def print_summary(results: Sequence[EvaluationResult]) -> None:
    """채점 결과를 사람이 읽기 쉬운 형태로 출력한다."""

    total_questions = len(results)
    correct_answers = sum(1 for result in results if result.is_correct)
    accuracy = (correct_answers / total_questions * 100.0) if total_questions else 0.0

    reasoning_failures = [
        result for result in results if result.reasoning_failure is not None or result.predicted_answer is None
    ]

    print("\n==================== 의료 벤치마크 요약 ====================")
    print(f"총 문항 수: {total_questions}")
    print(f"정답 수: {correct_answers}")
    print(f"정확도: {accuracy:.2f}%")

    if reasoning_failures:
        print("\n파싱/형식 실패 내역:")
        for result in reasoning_failures:
            reason = result.reasoning_failure or "예측 정답 없음"
            print(
                f"- [{result.index}] 예측={result.predicted_answer or 'N/A'} / 정답={result.correct_answer} / 사유={reason}"
            )
    else:
        print("\n파싱/형식 실패 내역: 없음")

    print("\n문항별 결과:")
    for result in results:
        status = "정답" if result.is_correct else "오답"
        predicted = result.predicted_answer or "N/A"
        print(f"- [{result.index}] {status} | 예측={predicted} | 정답={result.correct_answer}")


def save_results(results: Sequence[EvaluationResult], output_dir: str) -> None:
    """평가 결과를 별도 하위 폴더에 저장한다."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    total_questions = len(results)
    correct_answers = sum(1 for result in results if result.is_correct)
    accuracy = (correct_answers / total_questions * 100.0) if total_questions else 0.0

    summary_payload = {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": round(accuracy, 2),
        "reasoning_failures": [
            {
                "index": result.index,
                "predicted_answer": result.predicted_answer,
                "correct_answer": result.correct_answer,
                "reasoning_failure": result.reasoning_failure,
            }
            for result in results
            if result.reasoning_failure is not None or result.predicted_answer is None
        ],
    }

    detail_payload = [
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

    summary_path = target_dir / "medical_benchmark_summary.json"
    detail_path = target_dir / "medical_benchmark_details.json"

    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path.write_text(json.dumps(detail_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[INFO] 요약 저장: {summary_path}")
    print(f"[INFO] 상세 저장: {detail_path}")


def main() -> None:
    """스크립트 진입점."""

    args = parse_args()
    results = evaluate_dataset(MEDICAL_MCQ_DATA)
    print_summary(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()