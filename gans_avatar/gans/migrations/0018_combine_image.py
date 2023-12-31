# Generated by Django 4.2.5 on 2023-11-09 10:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("gans", "0017_rename_audio_inputs_audio_input"),
    ]

    operations = [
        migrations.CreateModel(
            name="Combine_Image",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("combined_image", models.ImageField(upload_to="images/combine/")),
                (
                    "combined_image_name",
                    models.CharField(blank=True, max_length=500, null=True),
                ),
            ],
        ),
    ]
