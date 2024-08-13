from django.urls import path
from . import views

# The path() function is passed four arguments, two of which are optional: route, view, kwargs, and name.
urlpatterns = [ 
    path('', views.index, name='index'),
]