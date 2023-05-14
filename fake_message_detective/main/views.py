from django.shortcuts import redirect, render
from .models import Message, Warning
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
        std.user='주녕'
        std.message_content = request.POST['message_content']
        std.message_sent_time = datetime.now()
        std.save()

    return redirect('/')

def inputInvalid(request):
    if request.method == 'POST':
        std =Warning()
        std.warning_valid=0
        std.save()

    return redirect('/')

def inputValid(request):
    if request.method == 'POST':
        std =Warning()
        std.warning_valid=1
        std.save()

    return redirect('/')