from .models import *
from rest_framework.serializers import ModelSerializer

class lawsSerailizer(ModelSerializer):
    class Meta:
        model = laws
        fields = "__all__"
