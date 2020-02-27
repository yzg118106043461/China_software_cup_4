from django.conf.urls import include, url
from django.contrib import admin
from disk import views as disk_views 

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    #url(r'^disk/$', disk_views.register),
    url(r'^$', disk_views.register),
]