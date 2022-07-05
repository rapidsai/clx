from clxquery import views
from django.conf.urls import re_path

urlpatterns = [re_path("clxquery/$", views.ExecuteClxQuery.as_view())]
