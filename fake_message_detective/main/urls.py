from django.urls import path

from . import views



urlpatterns = [
    path('', views.index, name='index'),
    path('messagedatas/', views.getMessageDatas, name="messagedatas"),
    path('inputform/', views.inputform, name='inputform'),
    path('inputInvalid/', views.inputInvalid, name='inputInvalid') ,
    path('inputValid/', views.inputValid, name='inputValid'),
    path('result/', views.classifyBanJon, name='classifyBanJon'),
    path('result2/', views.iconicSpell, name='iconicSpell'),
    path('findWarningMessage/', views.findWarningMessage, name='findWarningMessage'),
]
