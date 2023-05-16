from django.shortcuts import get_object_or_404, redirect, render
from .models import Message, Warning, User
# Create your views here.
def index(request):
    messageList = Message.objects.all()
    return render(request,'index.html',{'messageList':messageList})


from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import MessageSerializer

@api_view(['GET'])
def getMessageDatas(request):
    datas = Message.objects.all()
    serializer = MessageSerializer(datas, many=True)
    return Response(serializer.data)

from datetime import datetime

def inputform(request):
    if request.method == 'POST':
        std = Message()
        std.user=User.objects.get(user_name = '주녕')
        std.message_content = request.POST['message_content']
        std.message_sent_time = datetime.now()
        std.save()

    return redirect('/')

def inputInvalid(request):
    if request.method == 'POST':
        std =Warning()
        std.user = User.objects.get(user_name=request.POST['warning_message_user'])
        std.message = Message.objects.get(message_id=request.POST['warning_message_id'])
        std.warning_time=datetime.now()
        std.warning_valid=0
        std.save()


    return redirect('/')

def inputValid(request):
    if request.method == 'POST':
        std =Warning()
        std.user = User.objects.get(user_name=request.POST['warning_message_user'])
        std.message = Message.objects.get(message_id=request.POST['warning_message_id'])
        std.warning_time=datetime.now()
        std.warning_valid=1
        std.save()

    return redirect('/')
