# Generated by Django 4.2.5 on 2023-11-08 11:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("gans", "0009_avatar_background_delete_avatarandbackground"),
    ]

    operations = [
        migrations.AddField(
            model_name="avatar",
            name="image_name",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]