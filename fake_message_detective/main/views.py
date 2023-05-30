from django.shortcuts import get_object_or_404, redirect, render
from .models import Message, Warning, User
# Create your views here.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def index(request):
    messageList = Message.objects.all()
    return render(request, 'index.html', {'messageList': messageList})


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
        std = Warning.objects.get(message=request.POST['warning_message_id'])
        std.warning_valid=0
        std.save()

    return redirect('/')

def inputValid(request):
    if request.method == 'POST':
        std = Warning.objects.get(message=request.POST['warning_message_id'])
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
from .dataset import dataset_labels, dataset_sentences, phishing_keyword

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
    early_stopping = EarlyStopping()

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
    model.fit(padded_sequences, labels, epochs=10, callbacks=[early_stopping] )

    # 새로운 텍스트 예측
    # 모든 message_content 가져오기
    messages = Message.objects.all()
    # message_content 값만 가져오기
    new_texts = [spell_checker.check(message.message_content).checked for message in messages]
    
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding='post')
    predictions = model.predict(new_padded_sequences)

    count_ban = 0
    count_jon = 0

    for i, text in enumerate(new_texts):
        if predictions[i] < 0.5:
            print(f"'{text}'은(는) 반말입니다.")
            count_ban += 1
        else:
            print(f"'{text}'은(는) 존댓말입니다.")
            count_jon += 1

    user = User.objects.get(user_name='주녕')
    if count_ban > count_jon:
        user.accent_formal=0
    else:
        user.accent_formal=1
    user.save()
    return redirect('/')

##자주 틀리는 맞춤법 추출
from .hanspell import spell_checker
from collections import Counter
import string
   
def iconicSpell(request):
    # 문법 오류 교정
    def check_spelling_errors(text):
        spelled = spell_checker.check(text)
        checked_text = spelled.checked

        return checked_text

    # 맞춤법 틀린 부분 찾기 - 원본 텍스트와 교정 텍스트에서 다른 부분 비교
    def find_different_characters(text1, text2):
        differences = []

        # 두 문장의 길이 중 짧은 길이를 기준으로 순회
        min_length = min(len(text1), len(text2))
        for i in range(min_length):
            if text1[i] != text2[i]:
                differences.append((text2[i], text1[i]))

        # 길이가 다른 나머지 부분의 글자 추가
        if len(text1) > len(text2):
            differences.extend([(char, '') for char in text1[min_length:]])
        elif len(text1) < len(text2):
            differences.extend([('', char) for char in text2[min_length:]])

        return differences

    # 가장 많이 틀리는 맞춤법 추출
    def find_most_common_characters(differences, top_n):
        # 문장부호 제외
        punctuation = set(string.punctuation)
        differences = [(char1, char2) for char1, char2 in differences if char1 not in punctuation and char2 not in punctuation]
        
        counter = Counter(differences)
        most_common = counter.most_common(top_n)
        return [mc[0] for mc in most_common]

    messages = Message.objects.all()
    # message_content 값만 가져오기
    texts = [message.message_content for message in messages]

    all_differences = []

    for text in texts:
        # 맞춤법 오류 체크
        checked_text= check_spelling_errors(text)
        # 띄어쓰기 전처리를 하여 불필요한 결과값 없앰

        text=[c for c in text if c != ' ']
        checked_text=[c for c in checked_text if c != ' ']
        print("원본 텍스트:", text)
        print("교정된 텍스트:", checked_text)

        # 차이를 찾아내고 모든 차이를 합침
        differences = find_different_characters(text, checked_text)
        all_differences.extend(differences)

    print(all_differences)
    # 가장 많이 틀리는 맞춤법 상위 3개 추출
    top_3_most_common_characters = find_most_common_characters(all_differences, 3)

    print(top_3_most_common_characters)
    user = User.objects.get(user_name='주녕')
    user.accent_spell=str(top_3_most_common_characters)
    user.save()
    return redirect('/')

def findWarningMessage(request):
    messageList=Message.objects.all()
    message_info={}
    for message in messageList:
        message_info[message.message_id]=[]
    user = User.objects.get(user_name='주녕') 

    def is_formal_match():
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
        early_stopping = EarlyStopping()

        # 데이터 형식 맞추기
        padded_sequences = np.array(padded_sequences)
        labels = np.array(labels)

        # 데이터셋을 훈련 데이터와 테스트 데이터로 분리
        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

        # 모델 생성 및 훈련
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(len(word_index) + 1, 32, input_length=max_length),
            tf.keras.layers.Conv1D(64, 5, activation='relu'),
            tf.keras.layers.MaxPooling1D(4),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, callbacks=[early_stopping])

        # 테스트 데이터에 대한 예측
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # 정확도 평가
        accuracy = accuracy_score(y_test, y_pred)
        print("테스트 데이터 정확도:", accuracy)

        # 새로운 텍스트 예측
        new_texts = [spell_checker.check(message.message_content).checked for message in messageList]

        new_sequences = tokenizer.texts_to_sequences(new_texts)
        new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding='post')
        predictions = model.predict(new_padded_sequences)

        for i, text in enumerate(new_texts):
            if predictions[i] <= 0.5:
                is_formal = 0
            else:
                is_formal = 1

            print(text, ':', is_formal)
            print(is_formal, user.accent_formal)

            if is_formal != user.accent_formal:
                message_info[i+1].append(1)
            else:
                message_info[i+1].append(0)
                
    def spell_match():
        user_spell_tuple=eval(user.accent_spell)
        user_spell=[user_spell_tuple[i][1] for i in range(3)]
        origin_spell=[user_spell_tuple[i][0] for i in range(3)]

        new_texts = [message.message_content for message in messageList]
        for i, text in enumerate(new_texts):
            if any(char in text for char in origin_spell):
                if not any(char2 in text for char2 in user_spell):
                    message_info[i+1].append(1)
                else:
                    message_info[i+1].append(0)
            else:
                message_info[i+1].append(0)
    def keyword_match():
        new_texts = [message.message_content for message in messageList]
        for i, text in enumerate(new_texts):
            if any(char in text for char in phishing_keyword):       
                message_info[i+1].append(1)
            else:
                message_info[i+1].append(0)

    if len(messageList)>10:
        is_formal_match()
        spell_match()
        keyword_match()
        for message in messageList:
            w_message=Message.objects.get(message_id=message.message_id)

            if message_info[message.message_id].count(1) >= 2:
                w_message.is_warning=1
                if not Warning.objects.filter(message_id=message.message_id):
                    std =Warning()
                    std.message = Message.objects.get(message_id=message.message_id)
                    std.user = User.objects.get(user_name="주녕")
                    std.message = Message.objects.get(message_id=message.message_id)
                    std.warning_time=datetime.now()
                    std.warning_cause=''
                    for i in range(3):
                        if message_info[message.message_id][i]==1 and i==0:
                            std.warning_cause+='반말/존댓말 여부 다름,'
                        if message_info[message.message_id][i]==1 and i==1:
                            std.warning_cause+='자주 틀리는 맞춤법을 사용하지 않음,'
                        if message_info[message.message_id][i]==1 and i==2:
                            std.warning_cause+='피싱 메시지 위험 키워드 감지,'
                    std.save()
                else:
                    pass
            else:
                w_message.is_warning=0

            w_message.save()
            redirect("/")
    print(message_info)
    return redirect('/')