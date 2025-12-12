import uuid

from django.db import migrations, models


def generate_session_uuids(apps, schema_editor):
    ChatSession = apps.get_model('chatbot', 'ChatSession')
    for session in ChatSession.objects.all():
        session.session_uuid = uuid.uuid4()
        session.save(update_fields=['session_uuid'])


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0002_chatmessage_category_chatmessage_feedback_comment_and_more'),
    ]

    operations = [
        # 1) nullable, non-unique 컬럼 추가
        migrations.AddField(
            model_name='chatsession',
            name='session_uuid',
            field=models.UUIDField(db_index=True, editable=False, null=True, unique=False, verbose_name='세션 UUID'),
        ),
        # 2) 기존 레코드에 UUID 채우기
        migrations.RunPython(generate_session_uuids, reverse_code=migrations.RunPython.noop),
        # 3) 최종 스키마로 제약 변경 (not null + unique + default)
        migrations.AlterField(
            model_name='chatsession',
            name='session_uuid',
            field=models.UUIDField(db_index=True, default=uuid.uuid4, editable=False, unique=True, verbose_name='세션 UUID'),
        ),
    ]
