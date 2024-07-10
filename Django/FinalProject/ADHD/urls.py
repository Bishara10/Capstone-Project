from django.urls import path
from . import views

urlpatterns = [
    path('ADHD/', views.eyeTest, name='ADHD'), # Eye-Test
    path('', views.landingPage, name='landing'),
    path('landing', views.landingPage, name='home'),
    path('FAQ', views.FAQ, name='FAQ'),
    path('instructions', views.instructions, name='instructions'),
    path('video-configuration/', views.upload_video, name='upload'),
    path('vocal-test/', views.vocalTest , name='vocal'), # Vocal-Test
    path('questionnaire/', views.questionnaire, name='questionnaire'),
    path('audio-configuration/', views.upload_voice, name='voice_upload'),
    path('upload_answers/', views.upload_answers, name="answers"),
    path('results/', views.results, name='results'),
    path('processing/', views.processing, name='processing'),
    path('done/', views.clean, name='done')
]