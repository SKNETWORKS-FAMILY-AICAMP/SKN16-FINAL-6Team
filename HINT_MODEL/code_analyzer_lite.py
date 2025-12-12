"""
Code Analyzer Lite - Runpod용 경량 코드 분석기

Django 의존성 없이 독립적으로 실행되는 코드 분석 모듈입니다.
테스트 케이스 실행은 Django에서 처리하고 결과만 전달받습니다.
"""

import ast
import re
from typing import Dict, Any


def analyze_code_lite(code: str, problem_id: str = None) -> Dict[str, Any]:
    """
    코드의 정적 분석 수행 (경량 버전)

    Args:
        code: 분석할 Python 코드
        problem_id: 문제 ID (현재는 미사용)

    Returns:
        정적 메트릭 딕셔너리
    """
    metrics = {
        'syntax_errors': 0,
        'test_pass_rate': 0,  # Runpod에서는 계산 불가, Django에서 전달받아야 함
        'execution_time': 0,
        'memory_usage': 0,
        'code_quality_score': 50,  # 기본값
        'pep8_violations': 0,
        'cyclomatic_complexity': 1,
        'line_count': 0,
        'function_count': 0,
    }

    if not code or not code.strip():
        metrics['syntax_errors'] = 1
        metrics['code_quality_score'] = 0
        return metrics

    # 1. 문법 오류 체크
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        metrics['syntax_errors'] = 1
        metrics['code_quality_score'] = 0
        return metrics

    # 2. 기본 메트릭 계산
    lines = code.strip().split('\n')
    metrics['line_count'] = len(lines)

    # 3. 함수 개수
    function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    metrics['function_count'] = function_count

    # 4. 순환 복잡도 (간단 계산)
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    metrics['cyclomatic_complexity'] = complexity

    # 5. PEP8 위반 체크 (간단 버전)
    pep8_violations = 0

    for i, line in enumerate(lines, 1):
        # 라인 길이 체크 (79자 초과)
        if len(line) > 79:
            pep8_violations += 1

        # 탭 사용 체크
        if '\t' in line:
            pep8_violations += 1

        # 트레일링 공백 체크
        if line != line.rstrip():
            pep8_violations += 1

        # 연산자 주위 공백 체크 (간단)
        if re.search(r'[a-zA-Z0-9][=+\-*/][a-zA-Z0-9]', line):
            # 복합 연산자(==, +=, -= 등) 제외
            if not re.search(r'[=!<>+\-*/][=]', line):
                pep8_violations += 1

    metrics['pep8_violations'] = pep8_violations

    # 6. 코드 품질 점수 계산
    quality_score = 100

    # 복잡도 패널티
    if complexity > 10:
        quality_score -= min(30, (complexity - 10) * 3)

    # PEP8 패널티
    quality_score -= min(20, pep8_violations * 2)

    # 라인 수 패널티 (너무 길면)
    if metrics['line_count'] > 100:
        quality_score -= min(10, (metrics['line_count'] - 100) // 10)

    # 함수 사용 보너스
    if function_count > 0:
        quality_score += min(10, function_count * 2)

    # 주석 보너스
    comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
    if comment_lines > 0:
        quality_score += min(5, comment_lines)

    # 범위 제한
    metrics['code_quality_score'] = max(0, min(100, quality_score))

    return metrics


def check_syntax(code: str) -> Dict[str, Any]:
    """문법 오류만 빠르게 체크"""
    try:
        ast.parse(code)
        return {'valid': True, 'error': None}
    except SyntaxError as e:
        return {
            'valid': False,
            'error': {
                'line': e.lineno,
                'offset': e.offset,
                'message': str(e.msg) if hasattr(e, 'msg') else str(e)
            }
        }


def get_code_structure(code: str) -> Dict[str, Any]:
    """코드 구조 분석"""
    structure = {
        'functions': [],
        'classes': [],
        'imports': [],
        'global_variables': []
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return structure

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            structure['functions'].append({
                'name': node.name,
                'line': node.lineno,
                'args': [arg.arg for arg in node.args.args]
            })
        elif isinstance(node, ast.ClassDef):
            structure['classes'].append({
                'name': node.name,
                'line': node.lineno
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                structure['imports'].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            structure['imports'].append(f"{node.module}.{', '.join(a.name for a in node.names)}")

    return structure
