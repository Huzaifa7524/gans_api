# Generated by Django 4.2.5 on 2023-11-08 12:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("gans", "0011_alter_avatar_image_name"),
    ]

    operations = [
        migrations.AlterField(
            model_name="avatar",
            name="image_name",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]
