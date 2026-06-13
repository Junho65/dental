from django.db import migrations, models


PRICE_FIELD_MAP = {
    "price_unprc1": "unprc1",
    "price_unprc2": "unprc2",
    "price_unprc3": "unprc3",
    "price_unprc4": "unprc4",
    "price_unprc5": "unprc5",
    "price_unprc6": "unprc6",
}


def _parse_price(value):
    if value in (None, "", "0", "0.0"):
        return None
    try:
        parsed = int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_procedure_type(raw_value):
    normalized = str(raw_value or "").strip()
    if normalized == "수술":
        return "surgery"
    if normalized == "비수술":
        return "non_surgery"
    return ""


def _infer_surgery_role(kor_nm):
    normalized = str(kor_nm or "").strip()
    if not normalized:
        return ""
    return "secondary" if "제2의수술" in normalized else "primary"


def _has_disability_surcharge(kor_nm):
    normalized = str(kor_nm or "").strip()
    return any(token in normalized for token in ("장애인가산", "장애인 가산", "장애가산"))


def backfill_mdfeeitem_structured_fields(apps, schema_editor):
    MdFeeItem = apps.get_model("classifier", "MdFeeItem")

    for item in MdFeeItem.objects.all().iterator():
        raw_item = item.raw_item if isinstance(item.raw_item, dict) else {}
        for field_name, raw_key in PRICE_FIELD_MAP.items():
            setattr(item, field_name, _parse_price(raw_item.get(raw_key)))

        item.procedure_type = _normalize_procedure_type(raw_item.get("soprTpNm"))
        item.surgery_role = _infer_surgery_role(item.kor_nm)
        item.disability_surcharge = _has_disability_surcharge(item.kor_nm)
        item.save(
            update_fields=[
                *PRICE_FIELD_MAP.keys(),
                "procedure_type",
                "surgery_role",
                "disability_surcharge",
            ]
        )


class Migration(migrations.Migration):

    dependencies = [
        ("classifier", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="mdfeeitem",
            name="disability_surcharge",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="price_unprc1",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="price_unprc2",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="price_unprc3",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="price_unprc4",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="price_unprc5",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="price_unprc6",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="procedure_type",
            field=models.CharField(blank=True, max_length=16),
        ),
        migrations.AddField(
            model_name="mdfeeitem",
            name="surgery_role",
            field=models.CharField(blank=True, max_length=16),
        ),
        migrations.RunPython(backfill_mdfeeitem_structured_fields, migrations.RunPython.noop),
    ]
