from django.apps import AppConfig


class FsrpConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'FSRP'
    
    def ready(self):
        import FSRP.signals
