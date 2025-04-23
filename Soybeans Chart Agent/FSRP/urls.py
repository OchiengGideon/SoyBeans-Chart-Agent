from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns= [
     path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('login/', views.loginPage, name="login"),
    path('logout/', views.logout_user, name='logout'),
    path('register/',views.register_user, name='register'),
    path('profile/<str:pk>/', views.userProfile, name="user-profile"),
    path('contact/', views.contact, name='contact'),
    path('info/<str:pk>/', views.info, name="info"),
    path('create-info/', views.createInfo, name="create-info"),
    path('update-info/<str:pk>/', views.updateInfo, name="update-info"),
    path('delete-room/<str:pk>/', views.deleteRoom, name="delete-room"),
    path('delete-message/<str:pk>/', views.deleteMessage, name="delete-message"),
    path('index/',views.index, name='index'),
    path('update-user/', views.updateUser, name="update-user"),
    path('all-info/',views.allInfo, name="all-info"),
    path('topics/', views.topicsPage, name="topics"),
    path('activity/', views.activityPage, name="activity"),
    path('soil-details',views.soil_details, name='soil-details'),
    path('update-soil-details',views.update_soil_details, name='update-soil-details'),
    path('recommendation/', views.recommendation, name="recommendation"),
    path('get-weather/',views.get_weather,name="get_weather"),
    path('user-location/', views.user_location, name="user-location"),
    path('farm-profile/', views.farm_profile, name="farm-profile"),
    path('chart/', views.chart_view, name='chart'),
    path('ask', views.ask, name='ask')
]
   

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)