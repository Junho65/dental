from django.core.management.base import BaseCommand, CommandError

from django_app.classifier.mdfee import sync_mdfee_items


class Command(BaseCommand):
    help = "Fetch HIRA medical fee items for configured dental lesion mappings and store them in the DB."

    def add_arguments(self, parser):
        parser.add_argument("--service-key", default=None, help="Override DATAGO_KEY.")
        parser.add_argument("--timeout", type=float, default=None, help="HTTP timeout in seconds.")
        parser.add_argument(
            "--lesion-class",
            action="append",
            dest="lesion_classes",
            default=None,
            help="Only sync the specified lesion class. Repeat to sync multiple classes.",
        )

    def handle(self, *args, **options):
        try:
            stats = sync_mdfee_items(
                service_key=options["service_key"],
                timeout=options["timeout"],
                lesion_classes=options["lesion_classes"],
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Synced HIRA fees: "
                f"requests={stats.requested}, received={stats.received}, "
                f"saved={stats.saved}, skipped={stats.skipped}, errors={stats.errors}"
            )
        )
        if stats.error_messages:
            for message in stats.error_messages:
                self.stderr.write(self.style.WARNING(f"HIRA sync warning: {message}"))
