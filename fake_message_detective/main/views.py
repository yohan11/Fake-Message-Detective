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

#말투 학습 부분

## 반말/존댓말 구분
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from .dataset import dataset_labels, dataset_sentences

def classifyBanJon(request):
    # 반말과 존댓말 데이터 셋 준비
    sentences = dataset_sentences
    labels = dataset_labels  # 1: 반말, 0: 존댓말

    # 텍스트 데이터를 정수 시퀀스로 변환
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)

    # 시퀀스 패딩
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    #overfitting을 방지하기 위한 early stopping
    early_stopping = EarlyStopping(

    min_delta=0.001,   # 개선으로 간주되는 최소 변경 크기. 이 값만큼 개선이 없으면 Early Stopping 대상이 됩니다. 

    patience=20,          # Early Stopping이 실제로 학습을 중지하기 전에 몇 epoch를 기다릴지를 의미합니다. 

    restore_best_weights=True, # Early Stopping시 이전에 찾은 최적의 가중치값으로 복원합니다. 

)

    # 모델 생성
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(len(word_index) + 1, 16, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #데이터 형식 맞추기
    padded_sequences = np.array(padded_sequences)
    labels = np.array(labels)
    # 모델 훈련
    model.fit(padded_sequences, labels, epochs=30, callbacks=[early_stopping] )

    # 새로운 텍스트 예측
    # 모든 message_content 가져오기
    messages = Message.objects.all()
    # message_content 값만 가져오기
    new_texts = [message.message_content for message in messages]
    
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding='post')
    predictions = model.predict(new_padded_sequences)

    count_ban = 0
    count_jon = 0
    is_banjon = ''
    for i, text in enumerate(new_texts):
        if predictions[i] > 0.5:
            print(f"'{text}'은(는) 반말입니다.")
            count_ban += 1
        else:
            print(f"'{text}'은(는) 존댓말입니다.")
            count_jon += 1
    if count_ban > count_jon:
        is_banjon="해당 사용자는 반말을 더 많이 사용함"
    else:
        is_banjon="해당 사용자는 존댓말을 더 많이 사용함"

    return render(request,'result.html',{'ban_jon':is_banjon})

   

