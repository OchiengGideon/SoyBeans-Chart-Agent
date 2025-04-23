from django.contrib import admin

# Register your models here.
from .models import Info, Profile, Topic, Message, Soil

admin.site.register(Info)
admin.site.register(Topic)
admin.site.register(Message)
admin.site.register(Profile)
admin.site.register(Soil)