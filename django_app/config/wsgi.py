import os

from django.core.wsgi import get_wsgi_application

from django_app.config.env import load_env_file

load_env_file()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_app.config.settings")
application = get_wsgi_application()
