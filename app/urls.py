from django.urls import path
from . import views
from . import viewAPI

# The path() function is passed four arguments, two of which are optional: route, view, kwargs, and name.
urlpatterns = [ 
    path('', views.index, name='index'),
    path('video/', views.video, name='video'),

    # API
    # load objects
    path('api/objects/', viewAPI.Objects.as_view(), name='objects'),
    path('api/colors/', viewAPI.Colors.as_view(), name='colors'),

    # Query
    path('api/text_search_image/', viewAPI.TextSearchImage.as_view(), name='text_search_image'),
    path('api/image_query/', viewAPI.ImageQuery.as_view(), name='image_query'),
    path('api/filter_search/', viewAPI.FilterSearch.as_view(), name='filter_search'),
    path('api/tag_query/', viewAPI.TagQuery.as_view(), name='tag_query'),
    path('api/media_info_search/', viewAPI.MediaInfoVideo.as_view(), name='media_info_search'),

    # cluster
    path('api/cluster_frames/', viewAPI.ClusterFrames.as_view(), name='cluster_frames'),
    path('api/image_search_cluster/', viewAPI.ImageSearchCluster.as_view(), name='image_search_cluster'),
    
    # feelback
    path('api/feelback/', viewAPI.Feelback.as_view(), name='feelback'),

    # LLM
    path('api/llm/', viewAPI.LLM.as_view(), name='llm'),
    
    path('api/llm_chatbot/', viewAPI.LLMChatbot.as_view(), name='llm_chatbot'),
]