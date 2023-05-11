from rest_framework import serializers
from .models import Message

class MessageSerializer(serializers.ModelSerializer) :
    class Meta :
        model = Message        # product 모델 사용
        fields = '__all__'            # 모든 필드 포함