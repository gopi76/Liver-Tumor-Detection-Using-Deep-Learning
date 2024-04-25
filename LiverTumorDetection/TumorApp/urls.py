from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),	      
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
               path("UpdateProfileAction", views.UpdateProfileAction, name="UpdateProfileAction"),
               path("UpdateProfile.html", views.UpdateProfile, name="UpdateProfile"),
               path("DetectionAction", views.DetectionAction, name="DetectionAction"),
               path("Detection.html", views.Detection, name="Detection"),	       
]
