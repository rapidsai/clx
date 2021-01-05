from django.conf.urls import re_path
from clxquery import views

urlpatterns = [re_path("clxquery/$", views.ExecuteClxQuery.as_view())]