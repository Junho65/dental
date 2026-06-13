from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="MdFeeItem",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("keyword", models.CharField(max_length=100)),
                ("lesion_class", models.CharField(max_length=64)),
                ("treatment_name", models.CharField(max_length=120)),
                ("mdfee_cd", models.CharField(max_length=32)),
                ("mdfee_div_no", models.CharField(blank=True, max_length=32)),
                ("kor_nm", models.CharField(max_length=200)),
                ("pay_tp_nm", models.CharField(blank=True, max_length=130)),
                ("unit_price", models.PositiveIntegerField()),
                ("adtsta_dd", models.DateField()),
                ("cval_pnt", models.CharField(blank=True, max_length=40)),
                ("raw_item", models.JSONField(blank=True, default=dict)),
                ("synced_at", models.DateTimeField()),
            ],
        ),
        migrations.AddIndex(
            model_name="mdfeeitem",
            index=models.Index(fields=["lesion_class"], name="classifier__lesion__1ad91e_idx"),
        ),
        migrations.AddIndex(
            model_name="mdfeeitem",
            index=models.Index(fields=["keyword"], name="classifier__keyword_70ca9b_idx"),
        ),
        migrations.AddConstraint(
            model_name="mdfeeitem",
            constraint=models.UniqueConstraint(
                fields=("mdfee_cd", "adtsta_dd", "keyword", "lesion_class"),
                name="uniq_mdfee_item_per_keyword_lesion_date",
            ),
        ),
    ]
