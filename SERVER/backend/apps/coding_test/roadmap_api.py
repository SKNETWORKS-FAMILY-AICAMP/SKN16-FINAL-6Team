"""
로드맵, 설문조사, 뱃지, 목표 관련 API
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.utils import timezone
from .models import (
    UserSurvey, Roadmap, Badge, UserBadge,
    Goal, UserGoal, Problem, Submission
)
from .badge_logic import check_and_award_badges
import json
import random

# 설문조사 value → DB category 매핑 테이블
SURVEY_TO_DB_CATEGORY = {
    # 쉬운 카테고리
    "data_structures": ["자료구조 (기본)", "해시/맵"],
    "string": ["문자열"],
    "sorting": ["정렬"],
    "search": ["탐색", "브루트포스/백트래킹"],
    "implementation": ["입출력/기초"],
    "math": ["수학"],

    # 보통 카테고리
    "dp": ["DP (동적계획법)"],
    "graph_tree": ["그래프 (기본)", "트리"],
    "bitmask": ["고급 알고리즘"],

    # 어려운 카테고리
    "greedy": ["그리디"],
    "segment_tree": ["자료구조 (고급)"],
    "shortest_path": ["그래프 (고급)"],
    "mst": ["그래프 (고급)"],
    "network_flow": ["네트워크 플로우"],
    "advanced_ds": ["자료구조 (고급)", "고급 알고리즘"]
}

# 카테고리 난이도 분류
CATEGORY_DIFFICULTY = {
    # 쉬운 카테고리
    "data_structures": "쉬움",
    "string": "쉬움",
    "sorting": "쉬움",
    "search": "쉬움",
    "implementation": "쉬움",
    "math": "쉬움",

    # 보통 카테고리
    "dp": "보통",
    "graph_tree": "보통",
    "bitmask": "보통",

    # 어려운 카테고리
    "greedy": "어려움",
    "segment_tree": "어려움",
    "shortest_path": "어려움",
    "mst": "어려움",
    "network_flow": "어려움",
    "advanced_ds": "어려움"
}

# 경험 수준 + 최대 난이도별 Level 범위
LEVEL_RANGE_MAP = {
    ("beginner", "쉬움"): (1, 12),
    ("beginner", "보통"): (1, 14),
    ("beginner", "어려움"): (1, 16),
    ("intermediate", "쉬움"): (8, 16),
    ("intermediate", "보통"): (8, 18),
    ("intermediate", "어려움"): (8, 22),
    ("advanced", "쉬움"): (15, 22),
    ("advanced", "보통"): (15, 24),
    ("advanced", "어려움"): (15, 26),
}

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_survey(request):
    """설문조사 제출 및 로드맵 생성"""
    user = request.user

    # 설문조사 데이터
    survey_data = {
        'programming_experience': request.data.get('programming_experience'),
        'learning_goals': request.data.get('learning_goals', []),
        'interested_topics': request.data.get('interested_topics', []),
        'weak_topics': request.data.get('weak_topics', []),
        'target_type': request.data.get('target_type', 'daily'),
        'daily_problem_goal': request.data.get('daily_problem_goal', 2),
        'weekly_problem_goal': request.data.get('weekly_problem_goal', 14)
    }

    # 이미 설문조사가 있으면 업데이트, 없으면 생성
    survey, created = UserSurvey.objects.update_or_create(
        user=user,
        defaults=survey_data
    )

    # 기존 활성화된 로드맵 비활성화
    Roadmap.objects.filter(user=user, is_active=True).update(is_active=False)

    # 로드맵 생성 (자동으로 is_active=True)
    result = generate_roadmap(user, survey)

    # 에러 처리
    if not result.get('success', True):
        return Response(result, status=status.HTTP_400_BAD_REQUEST)

    roadmap = result['roadmap']

    # 기본 목표 생성 (기존 목표는 유지)
    create_default_goals(user, survey)

    message = '설문조사가 완료되었고 로드맵이 생성되었습니다.' if created else '설문조사가 업데이트되었고 로드맵이 재생성되었습니다.'

    return Response({
        'success': True,
        'message': message,
        'data': {
            'survey_id': survey.id,
            'roadmap_id': roadmap.id,
            'total_problems': len(roadmap.recommended_problems)
        }
    })


def generate_roadmap(user, survey):
    """설문조사 기반 로드맵 생성 (새로운 로직)"""

    # Step 0: 사용자 목표 기반 최소 문제 수 계산
    target_count = survey.daily_problem_goal * 7 if survey.target_type == 'daily' else survey.weekly_problem_goal
    MIN_ROADMAP_SIZE = target_count  # 사용자 목표가 최소값

    # Step 1: 선택한 카테고리 합치기
    selected_survey_categories = survey.interested_topics + survey.weak_topics

    if not selected_survey_categories:
        return {
            'success': False,
            'message': '관심 분야를 최소 1개 이상 선택해주세요.'
        }

    # Step 2: 최대 난이도 찾기
    max_difficulty = "쉬움"
    for survey_cat in selected_survey_categories:
        difficulty = CATEGORY_DIFFICULTY.get(survey_cat, "쉬움")
        if difficulty == "어려움":
            max_difficulty = "어려움"
            break
        elif difficulty == "보통" and max_difficulty == "쉬움":
            max_difficulty = "보통"

    # Step 3: Level 범위 결정
    min_level, max_level = LEVEL_RANGE_MAP.get(
        (survey.programming_experience, max_difficulty),
        (1, 12)
    )

    # Step 4: 설문조사 value를 DB category로 변환
    db_categories = []
    interested_db_cats = []
    weak_db_cats = []

    for survey_cat in survey.interested_topics:
        cats = SURVEY_TO_DB_CATEGORY.get(survey_cat, [])
        db_categories.extend(cats)
        interested_db_cats.extend(cats)

    for survey_cat in survey.weak_topics:
        cats = SURVEY_TO_DB_CATEGORY.get(survey_cat, [])
        db_categories.extend(cats)
        weak_db_cats.extend(cats)

    # 중복 제거
    db_categories = list(set(db_categories))

    if not db_categories:
        return {
            'success': False,
            'message': '선택한 카테고리에 해당하는 문제를 찾을 수 없습니다.'
        }

    # Step 5: 문제 필터링 및 가중치 점수 부여
    problems = Problem.objects.filter(
        level__gte=min_level,
        level__lte=max_level,
        category__in=db_categories
    )

    scored_problems = []
    mid_level = (min_level + max_level) // 2

    for problem in problems:
        score = 0

        # 관심 분야: +15점
        if problem.category in interested_db_cats:
            score += 15

        # 보완 분야: +10점
        if problem.category in weak_db_cats:
            score += 10

        # Level 기반 가중치: +5점
        if survey.programming_experience == 'beginner' and problem.level <= mid_level:
            score += 5
        elif survey.programming_experience == 'advanced' and problem.level >= mid_level:
            score += 5

        scored_problems.append({
            'problem': problem,
            'score': score,
            'level': problem.level,
            'problem_id': problem.problem_id
        })

    # 점수 내림차순 → Level 오름차순 → problem_id 오름차순 정렬
    scored_problems.sort(key=lambda x: (-x['score'], x['level'], x['problem_id']))

    # 조건에 맞는 모든 문제 선택 (MAX 제한 없음)
    selected_problems = scored_problems

    # Step 6: 문제 수 부족 시 확장
    if len(selected_problems) < MIN_ROADMAP_SIZE:
        # 전략 1: Level 범위 ±3 확장
        extended_problems = Problem.objects.filter(
            level__gte=max(1, min_level - 3),
            level__lte=min(26, max_level + 3),
            category__in=db_categories
        ).exclude(
            problem_id__in=[p['problem_id'] for p in selected_problems]
        ).order_by('level', 'problem_id')

        for ext_prob in extended_problems:
            selected_problems.append({
                'problem': ext_prob,
                'score': 0,
                'level': ext_prob.level,
                'problem_id': ext_prob.problem_id
            })
            if len(selected_problems) >= MIN_ROADMAP_SIZE:
                break

        # 전략 2: 여전히 부족하면 category 무시
        if len(selected_problems) < MIN_ROADMAP_SIZE:
            fallback_problems = Problem.objects.filter(
                level__gte=min_level,
                level__lte=max_level
            ).exclude(
                problem_id__in=[p['problem_id'] for p in selected_problems]
            ).order_by('level', 'problem_id')

            for fb_prob in fallback_problems:
                selected_problems.append({
                    'problem': fb_prob,
                    'score': 0,
                    'level': fb_prob.level,
                    'problem_id': fb_prob.problem_id
                })
                if len(selected_problems) >= MIN_ROADMAP_SIZE:
                    break

    # Step 7: 여전히 부족하면 에러
    if len(selected_problems) < MIN_ROADMAP_SIZE:
        return {
            'success': False,
            'message': f'생성할 문제 수가 부족합니다 (현재 {len(selected_problems)}개). 관심 분야를 더 선택하거나 경험 수준을 조정해주세요.',
            'available_count': len(selected_problems),
            'suggestion': '다른 카테고리를 추가로 선택하시면 더 많은 문제를 추천받을 수 있습니다.'
        }

    # Step 8: 로드맵 생성
    recommended_problem_ids = [p['problem_id'] for p in selected_problems]

    roadmap = Roadmap.objects.create(
        user=user,
        recommended_problems=recommended_problem_ids,
        current_step=0,
        is_active=True
    )

    return {
        'success': True,
        'roadmap': roadmap
    }


def create_default_goals(user, survey):
    """기본 목표 생성"""
    # 매일 로그인 목표
    daily_login_goal = Goal.objects.get_or_create(
        goal_type='daily_login',
        defaults={
            'name': '매일 로그인하기',
            'description': '7일 연속으로 로그인하세요',
            'target_value': 7
        }
    )[0]

    UserGoal.objects.get_or_create(
        user=user,
        goal=daily_login_goal
    )

    # 매일 문제 풀기 목표
    daily_problem_goal = Goal.objects.get_or_create(
        goal_type='daily_problem',
        defaults={
            'name': '매일 문제 풀기',
            'description': f'매일 {survey.daily_problem_goal}개씩 문제를 푸세요',
            'target_value': survey.daily_problem_goal
        }
    )[0]

    UserGoal.objects.get_or_create(
        user=user,
        goal=daily_problem_goal
    )

    # 주간 문제 목표
    weekly_target = survey.weekly_problem_goal if survey.target_type == 'weekly' else 10
    weekly_goal = Goal.objects.get_or_create(
        goal_type='weekly_problems',
        defaults={
            'name': '주간 문제 해결',
            'description': f'이번 주에 {weekly_target}개의 문제를 해결하세요',
            'target_value': weekly_target
        }
    )[0]

    UserGoal.objects.get_or_create(
        user=user,
        goal=weekly_goal
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_roadmap(request):
    """사용자의 활성화된 로드맵 조회"""
    user = request.user

    # 활성화된 로드맵 확인
    roadmap = Roadmap.objects.filter(user=user, is_active=True).first()
    if not roadmap:
        return Response({
            'success': False,
            'message': '설문조사를 먼저 완료해주세요.'
        }, status=status.HTTP_404_NOT_FOUND)

    # 추천 문제 정보 가져오기
    problems = Problem.objects.filter(
        problem_id__in=roadmap.recommended_problems
    ).values('problem_id', 'title', 'step_title', 'level', 'tags')

    # 순서 유지
    problem_dict = {p['problem_id']: p for p in problems}
    ordered_problems = [
        problem_dict[pid] for pid in roadmap.recommended_problems
        if pid in problem_dict
    ]

    return Response({
        'success': True,
        'data': {
            'roadmap': {
                'id': roadmap.id,
                'current_step': roadmap.current_step,
                'recommended_problems': roadmap.recommended_problems,
                'progress_percentage': roadmap.progress_percentage,
                'created_at': roadmap.created_at,
                'updated_at': roadmap.updated_at,
                'is_active': roadmap.is_active
            },
            'problems': ordered_problems
        }
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_roadmaps(request):
    """사용자의 모든 로드맵 조회"""
    user = request.user

    roadmaps = Roadmap.objects.filter(user=user).order_by('-is_active', '-created_at')

    roadmaps_data = []
    for roadmap in roadmaps:
        # 각 로드맵의 문제 개수 및 진행률 계산
        total_problems = len(roadmap.recommended_problems)
        roadmaps_data.append({
            'id': roadmap.id,
            'current_step': roadmap.current_step,
            'total_problems': total_problems,
            'progress_percentage': roadmap.progress_percentage,
            'is_active': roadmap.is_active,
            'created_at': roadmap.created_at,
            'updated_at': roadmap.updated_at
        })

    return Response({
        'success': True,
        'data': roadmaps_data
    })


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_roadmap(request, roadmap_id):
    """로드맵 삭제"""
    user = request.user

    try:
        roadmap = Roadmap.objects.get(id=roadmap_id, user=user)
    except Roadmap.DoesNotExist:
        return Response({
            'success': False,
            'message': '로드맵을 찾을 수 없습니다.'
        }, status=status.HTTP_404_NOT_FOUND)

    # 활성화된 로드맵을 삭제하는 경우
    was_active = roadmap.is_active
    roadmap.delete()

    # 삭제된 로드맵이 활성화 상태였다면, 가장 최근 로드맵을 활성화
    if was_active:
        latest_roadmap = Roadmap.objects.filter(user=user).order_by('-created_at').first()
        if latest_roadmap:
            latest_roadmap.is_active = True
            latest_roadmap.save()

    return Response({
        'success': True,
        'message': '로드맵이 삭제되었습니다.'
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def activate_roadmap(request, roadmap_id):
    """로드맵 활성화 (선택)"""
    user = request.user

    try:
        roadmap = Roadmap.objects.get(id=roadmap_id, user=user)
    except Roadmap.DoesNotExist:
        return Response({
            'success': False,
            'message': '로드맵을 찾을 수 없습니다.'
        }, status=status.HTTP_404_NOT_FOUND)

    # 모든 로드맵 비활성화
    Roadmap.objects.filter(user=user).update(is_active=False)

    # 선택한 로드맵 활성화
    roadmap.is_active = True
    roadmap.save()

    return Response({
        'success': True,
        'message': '로드맵이 활성화되었습니다.',
        'data': {
            'roadmap_id': roadmap.id,
            'is_active': roadmap.is_active
        }
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_badges(request):
    """모든 뱃지 조회"""
    badges = Badge.objects.all().values('id', 'badge_type', 'name', 'description', 'icon')

    return Response({
        'success': True,
        'data': list(badges)
    })


def initialize_badges():
    """DB에 뱃지가 없으면 초기화"""
    if Badge.objects.count() > 0:
        return

    # 뱃지 정의 (badge_type, name, description, icon)
    badges_data = [
        # 기본 배지
        ('hello_world', 'Hello World!', '첫 코드 제출', '👋'),
        ('first_problem', '첫 걸음', '첫 번째 문제 해결', '🎯'),
        ('problems_10', '열정가', '문제 10개 해결', '🔥'),
        ('problems_50', '도전자', '문제 50개 해결', '💪'),
        ('problems_100', '백문불여일견', '문제 100개 해결', '🏆'),

        # 연속 출석/문제 풀기
        ('attendance_3days', '3일 출석', '3일 연속 출석', '📅'),
        ('problem_streak_7', '일주일 챌린지', '7일 연속 문제 풀기', '🗓️'),
        ('problem_streak_30', '한 달 챌린지', '30일 연속 문제 풀기', '📆'),

        # 문법 마스터 시리즈
        ('syntax_perfect', '문법 나치', '문법 오류 평균 0개', '✨'),
        ('syntax_careful', '꼼꼼 감정사', '문법 오류 평균 1-2개', '🔍'),
        ('syntax_racer', '오타 레이서', '문법 오류 평균 3-4개', '🏎️'),
        ('syntax_typo_monster', '타이핑 괴물', '문법 오류 평균 5-6개', '⌨️'),
        ('korean_grammar', '유사 한국인', '문법 오류 평균 7개 이상', '🇰🇷'),

        # 코딩 실력 시리즈
        ('skill_genius', '코딩 천재', '평균 알고리즘 패턴 일치도 80% 이상', '🧠'),
        ('skill_master', '알고리즘 마스터', '평균 알고리즘 패턴 일치도 60-79%', '🎓'),
        ('skill_steady', '꾸준러', '평균 알고리즘 패턴 일치도 40-59%', '🐢'),
        ('skill_newbie', '코딩 새싹', '힌트를 1번 이상 요청', '🌱'),

        # 논리 사고 시리즈
        ('logic_king', '로직 킹', '평균 엣지 케이스 처리 4점 이상', '👑'),
        ('logic_trial_error', '시행착오의 달인', '평균 엣지 케이스 처리 2.5-4점', '🔄'),
        ('logic_action_first', '일단 고', '평균 엣지 케이스 처리 2.5점 미만', '🚀'),

        # 특수 배지
        ('no_hint_10', '자력갱생', '10개 문제 힌트 없이 해결', '💡'),
        ('perfect_coder', '퍼펙트 코더', '완벽한 코드 작성자', '⭐'),
        ('unbreakable', '불굴의 의지', '한 문제에 5번 이상 힌트 요청', '🔥'),
        ('hint_collector', '힌트 수집가', '총 30회 이상 힌트 요청', '📚'),
        ('persistence_king', '끈기왕', '어려운 문제 포기하지 않기', '💎'),
        ('all_rounder', '만능 개발자', '모든 지표 평균 이상', '🌟'),

        # 기타 배지
        ('streak_7', '7일 연속 로그인', '7일 연속 로그인', '🔑'),
        ('perfect_score', '만점왕', '모든 테스트 케이스 통과', '💯'),
        ('all_easy', 'Easy 정복자', '모든 쉬운 문제 해결', '🎮'),
        ('speed_master', '스피드 마스터', '빠른 문제 해결', '⚡'),
        ('night_owl', '야행성', '자정 이후 문제 풀기', '🦉'),
        ('button_mania', '버튼 마니아', '실행 버튼 50회 이상', '🔘'),
    ]

    for badge_type, name, description, icon in badges_data:
        Badge.objects.get_or_create(
            badge_type=badge_type,
            defaults={
                'name': name,
                'description': description,
                'icon': icon
            }
        )
    print(f'[Badge] Initialized {len(badges_data)} badges')


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_badges(request):
    """사용자가 획득한 뱃지 조회 (진행 상황 포함)"""
    from .badge_logic import get_user_badge_progress, BADGE_CONDITIONS

    user = request.user

    # DB에 뱃지가 없으면 초기화
    initialize_badges()

    # 모든 뱃지 가져오기
    all_badges = Badge.objects.all()

    # 사용자가 획득한 뱃지
    user_badges = UserBadge.objects.filter(user=user).select_related('badge')
    earned_badge_ids = {ub.badge.id: ub.earned_at for ub in user_badges}

    # 진행 상황 계산
    progress = get_user_badge_progress(user)

    badges_data = []
    for badge in all_badges:
        badge_info = {
            'badge_id': badge.id,
            'badge_type': badge.badge_type,
            'name': badge.name,
            'description': badge.description,
            'icon': badge.icon,
            'earned': badge.id in earned_badge_ids,
            'earned_at': earned_badge_ids.get(badge.id),
        }

        # 진행 상황 추가
        if badge.badge_type in progress:
            prog = progress[badge.badge_type]
            badge_info['progress'] = {
                'current': prog['current'],
                'target': prog['target'],
                'percentage': prog['percentage'],
                'condition_description': prog['condition_description'],
                'condition_type': prog['condition_type']
            }
        elif badge.badge_type in BADGE_CONDITIONS:
            # BADGE_CONDITIONS에는 있지만 progress에 없는 경우
            desc, cond_type, target = BADGE_CONDITIONS[badge.badge_type]
            badge_info['progress'] = {
                'current': 0,
                'target': target,
                'percentage': 0,
                'condition_description': desc,
                'condition_type': cond_type
            }
        else:
            # 조건 정의가 없는 뱃지
            badge_info['progress'] = {
                'current': 0,
                'target': 0,
                'percentage': 0,
                'condition_description': badge.condition_description or badge.description,
                'condition_type': badge.condition_type or 'special'
            }

        badges_data.append(badge_info)

    return Response({
        'success': True,
        'data': badges_data
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_goals(request):
    """사용자 목표 진행 상황 조회"""
    user = request.user
    user_goals = UserGoal.objects.filter(user=user).select_related('goal')

    goals_data = [
        {
            'id': ug.id,
            'goal_id': ug.goal.id,
            'goal_name': ug.goal.name,
            'goal_description': ug.goal.description,
            'goal_type': ug.goal.goal_type,
            'target_value': ug.goal.target_value,
            'current_value': ug.current_value,
            'is_completed': ug.is_completed,
            'progress_percentage': ug.progress_percentage,
            'started_at': ug.started_at,
            'completed_at': ug.completed_at
        }
        for ug in user_goals
    ]

    return Response({
        'success': True,
        'data': goals_data
    })


# check_and_award_badges는 badge_logic.py에 정의되어 있음 (중복 제거)


def update_user_goals(user):
    """사용자 목표 진행 상황 업데이트"""
    today = timezone.now().date()

    # 매일 문제 풀기 목표 업데이트
    daily_problem_goals = UserGoal.objects.filter(
        user=user,
        goal__goal_type='daily_problem',
        is_completed=False
    )

    for goal in daily_problem_goals:
        # 오늘 해결한 문제 수
        today_submissions = Submission.objects.filter(
            user=user,
            result='success',
            created_at__date=today
        ).values('problem_id').distinct().count()

        goal.current_value = today_submissions
        if goal.current_value >= goal.goal.target_value:
            goal.is_completed = True
            goal.completed_at = timezone.now()
        goal.save()

    # 주간 문제 목표 업데이트
    weekly_goals = UserGoal.objects.filter(
        user=user,
        goal__goal_type='weekly_problems',
        is_completed=False
    )

    for goal in weekly_goals:
        # 이번 주 해결한 문제 수
        week_start = today - timezone.timedelta(days=today.weekday())
        weekly_count = Submission.objects.filter(
            user=user,
            result='success',
            created_at__date__gte=week_start
        ).values('problem_id').distinct().count()

        goal.current_value = weekly_count
        if goal.current_value >= goal.goal.target_value:
            goal.is_completed = True
            goal.completed_at = timezone.now()
        goal.save()
