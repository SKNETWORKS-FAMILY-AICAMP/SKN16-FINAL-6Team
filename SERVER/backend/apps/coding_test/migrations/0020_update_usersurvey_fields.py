# Generated manually for updating UserSurvey model fields

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('coding_test', '0019_add_coh_fields'),
    ]

    operations = [
        # Remove preferred_difficulty field
        migrations.RemoveField(
            model_name='usersurvey',
            name='preferred_difficulty',
        ),
        # Add weak_topics field
        migrations.AddField(
            model_name='usersurvey',
            name='weak_topics',
            field=models.JSONField(
                default=list,
                help_text='보완하고 싶은 주제 리스트 (예: ["dp", "greedy"])',
                verbose_name='보완하고 싶은 분야'
            ),
        ),
        # Add target_type field
        migrations.AddField(
            model_name='usersurvey',
            name='target_type',
            field=models.CharField(
                choices=[('daily', '하루 목표'), ('weekly', '주간 목표')],
                default='daily',
                max_length=10,
                verbose_name='목표 타입'
            ),
        ),
        # Add weekly_problem_goal field
        migrations.AddField(
            model_name='usersurvey',
            name='weekly_problem_goal',
            field=models.IntegerField(
                default=14,
                help_text='일주일에 풀고 싶은 문제 개수 (7-70)',
                verbose_name='주간 목표 문제 수'
            ),
        ),
        # Update interested_topics help_text
        migrations.AlterField(
            model_name='usersurvey',
            name='interested_topics',
            field=models.JSONField(
                default=list,
                help_text='관심 있는 주제 리스트 (예: ["dp", "graph_tree", "greedy"])',
                verbose_name='관심 분야'
            ),
        ),
        # Update daily_problem_goal default and help_text
        migrations.AlterField(
            model_name='usersurvey',
            name='daily_problem_goal',
            field=models.IntegerField(
                default=2,
                help_text='하루에 풀고 싶은 문제 개수 (1-10)',
                verbose_name='하루 목표 문제 수'
            ),
        ),
    ]
