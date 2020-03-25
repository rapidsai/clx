from django.urls import path
from clxquery import views

urlpatterns = [path("clxquery/<str:query>", views.run_query)]
