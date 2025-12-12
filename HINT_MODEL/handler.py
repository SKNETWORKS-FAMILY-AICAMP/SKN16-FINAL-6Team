"""
Runpod Serverless Handler for Hint Service

이 파일은 Runpod Serverless에서 실행되는 힌트 서비스의 엔트리포인트입니다.
Django와 분리되어 독립적으로 힌트 생성 로직을 실행합니다.

API 계약:
- Input: {
    "type": "hint" | "ping",
    "problem_id": str,
    "user_code": str,
    "star_count": int,  # Django에서 조회해서 전달
    "preset": str,
    "custom_components": dict,
    "previous_hints": list,
    "problem_data": dict  # 문제 정보 (title, description, solutions)
  }
- Output: {
    "success": bool,
    "data": { ... hint result ... } | null,
    "error": str | null
  }
"""

import runpod
from hint_core import generate_hint


def handler(event):
    """
    Runpod Serverless 핸들러 함수

    모든 요청은 이 함수를 통해 처리됩니다.
    """
    try:
        input_data = event.get("input", {})
        request_type = input_data.get("type", "hint")

        # Ping 요청 (Keep-Alive용)
        if request_type == "ping":
            return {
                "success": True,
                "data": {"status": "alive", "message": "Hint service is running"},
                "error": None
            }

        # 힌트 생성 요청
        if request_type == "hint":
            # 필수 파라미터 검증
            problem_id = input_data.get("problem_id")
            user_code = input_data.get("user_code", "")
            star_count = input_data.get("star_count", 0)
            preset = input_data.get("preset", "초급")
            custom_components = input_data.get("custom_components", {})
            previous_hints = input_data.get("previous_hints", [])
            problem_data = input_data.get("problem_data", {})

            if not problem_id:
                return {
                    "success": False,
                    "data": None,
                    "error": "problem_id is required"
                }

            # 힌트 생성
            result = generate_hint(
                problem_id=problem_id,
                user_code=user_code,
                star_count=star_count,
                preset=preset,
                custom_components=custom_components,
                previous_hints=previous_hints,
                problem_data=problem_data
            )

            return {
                "success": True,
                "data": result,
                "error": None
            }

        # 알 수 없는 요청 타입
        return {
            "success": False,
            "data": None,
            "error": f"Unknown request type: {request_type}"
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }


# Runpod Serverless 시작점
runpod.serverless.start({"handler": handler})
