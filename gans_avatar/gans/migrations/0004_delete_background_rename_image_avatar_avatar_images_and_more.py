# Generated by Django 4.2.7 on 2023-11-07 10:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gans', '0003_background'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Background',
        ),
        migrations.RenameField(
            model_name='avatar',
            old_name='image',
            new_name='avatar_images',
        ),
        migrations.RemoveField(
            model_name='avatar',
            name='name',
        ),
        migrations.AddField(
            model_name='avatar',
            name='background_images',
            field=models.ImageField(default=1, upload_to='images/bg/'),
            preserve_default=False,
        ),
    ]
