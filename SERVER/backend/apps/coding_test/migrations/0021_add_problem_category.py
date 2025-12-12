# Generated manually for adding category field to Problem model

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('coding_test', '0020_update_usersurvey_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='problem',
            name='category',
            field=models.CharField(
                blank=True,
                db_index=True,
                max_length=100,
                verbose_name='카테고리'
            ),
        ),
    ]
