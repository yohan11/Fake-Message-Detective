from django.contrib import admin
from .models import Message, Warning, User
# Register your models here.

admin.site.register(Message)
admin.site.register(Warning)
admin.site.register(User)
