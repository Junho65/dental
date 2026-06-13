from django.db import models


class MdFeeItem(models.Model):
    keyword = models.CharField(max_length=100)
    lesion_class = models.CharField(max_length=64)
    treatment_name = models.CharField(max_length=120)
    mdfee_cd = models.CharField(max_length=32)
    mdfee_div_no = models.CharField(max_length=32, blank=True)
    kor_nm = models.CharField(max_length=200)
    pay_tp_nm = models.CharField(max_length=130, blank=True)
    unit_price = models.PositiveIntegerField()
    price_unprc1 = models.PositiveIntegerField(null=True, blank=True)
    price_unprc2 = models.PositiveIntegerField(null=True, blank=True)
    price_unprc3 = models.PositiveIntegerField(null=True, blank=True)
    price_unprc4 = models.PositiveIntegerField(null=True, blank=True)
    price_unprc5 = models.PositiveIntegerField(null=True, blank=True)
    price_unprc6 = models.PositiveIntegerField(null=True, blank=True)
    adtsta_dd = models.DateField()
    cval_pnt = models.CharField(max_length=40, blank=True)
    procedure_type = models.CharField(max_length=16, blank=True)
    surgery_role = models.CharField(max_length=16, blank=True)
    disability_surcharge = models.BooleanField(default=False)
    raw_item = models.JSONField(default=dict, blank=True)
    synced_at = models.DateTimeField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["mdfee_cd", "adtsta_dd", "keyword", "lesion_class"],
                name="uniq_mdfee_item_per_keyword_lesion_date",
            )
        ]
        indexes = [
            models.Index(fields=["lesion_class"]),
            models.Index(fields=["keyword"]),
        ]

    def __str__(self) -> str:
        return f"{self.lesion_class} / {self.keyword} / {self.kor_nm}"
