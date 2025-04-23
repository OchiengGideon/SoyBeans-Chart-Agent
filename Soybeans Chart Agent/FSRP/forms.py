from django.forms import ModelForm
from django import forms
from .models import Info, User


class InfoForm(ModelForm):
  class Meta:
    model = Info
    fields = '__all__'
    exclude = ['host', 'participants']


