import datetime
import json
from django.conf import settings
from django.http import HttpResponse, HttpResponseNotAllowed, HttpResponseServerError, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
import requests
import time
from django.utils import timezone
from . tokens import generate_token
from django.contrib import messages
from .models import Info,Topic,Message, Profile, Soil
from .forms import InfoForm
from django.views.decorators.csrf import csrf_exempt
import os
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .models import FarmerQuery, Reminder
from .agents import (
    QueryExtractorAgent, InfoRetrievalAgent,
    MemoryAgent, ProfileAgent,
    ContextualAdviceAgent, ReminderAgent
)
import ast



@csrf_exempt  # in production, remove and use proper CSRF tokens in JS
@login_required(login_url='login')
def ask(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    raw_query = request.POST.get('user_input', '').strip()
    user = request.user

    # Initialize agents
    extractor = QueryExtractorAgent()
    retriever = InfoRetrievalAgent()
    memory    = MemoryAgent(user)
    profile   = ProfileAgent()
    advisor   = ContextualAdviceAgent(extractor, retriever, memory, profile)

    # Generate response
    advice = advisor.advise(raw_query)

    # Save the query and bot's response
    FarmerQuery.objects.create(
        user=user,
        query_text=raw_query,
        response_text=json.dumps(advice),
        timestamp=timezone.now()
    )

    print("advice+++++++++++++++++++++++++", advice)
    print(type(advice))
    advice = json.loads(advice)

    # Return only relevant fields (excluding chart)
    return JsonResponse({
        "response": {
            "answer": advice.get("answer"),
            "recommendations": advice.get("recommendations", []),
            "location": advice.get("location"),
        }
    })

# # Create your views here.
# def extract_text_from_pdfs(folder_path):
#     texts = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(folder_path, filename)
#             doc = fitz.open(file_path)
#             pdf_text = ""
#             for page in doc:
#                 pdf_text += page.get_text()
#             texts.append(pdf_text)
#     return " ".join(texts)

# model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# folder_path = 'pdfs'
# def preprocess_text(text, chunk_size=500, overlap=100):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i+chunk_size])
#         if len(chunk) > 100:  # ensure minimum info per chunk
#             chunks.append(chunk)
#     return chunks

# def embed_chunks(chunks):
#     embeddings = model.encode(chunks, convert_to_tensor=True)
#     return embeddings

# def search_information(query, text_chunks, chunk_embeddings):
#     query_embedding = model.encode([query], convert_to_tensor=True)
#     similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
#     best_idx = np.argmax(similarities)
#     best_score = similarities[best_idx]

#     if best_score < 0.4:  # adjust threshold as needed
#         return "Sorry, I couldn't find relevant info."
#     return text_chunks[best_idx] 


# raw_text = extract_text_from_pdfs("pdfs")
# text_chunks = preprocess_text(raw_text)  # from earlier
# chunk_embeddings = embed_chunks(text_chunks) 


def loginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = User.objects.get(username=username)
        except:
            messages.error(request, 'User does not exist')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, 'Username OR password does not exit')

    context = {}
    return render(request, 'Login.html', context)


def logout_user(request):
    logout(request)
    messages.success(request, "You have been loged out...")
    return redirect('home')

def home(request):
    #check if user is loging in
    q = request.GET.get('q') if request.GET.get('q') != None else ''

    infos = Info.objects.filter(
        Q(topic__name__icontains=q) |
        Q(name__icontains=q) |
        Q(description__icontains=q)
    )

    topics = Topic.objects.all()[0:5]
    room_count = infos.count()
    room_messages = Message.objects.filter(
        Q(info__topic__name__icontains=q))[0:3]

    context = {'infos': infos, 'topics': topics,
               'room_count': room_count, 'room_messages': room_messages}
    return render(request, 'FSRP/home.html', context)


def register_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        fname = request.POST['first_name']
        lname = request.POST['last_name']
        email = request.POST['email']
        pass1 = request.POST['password1']
        pass2 = request.POST['password2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('register_user')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('register_user')
        
        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('register_user')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('register_user')
        
        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!!")
            return redirect('register_user')
        
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        login(request, myuser)
        
        return redirect('index')
    
    return render(request, 'FSRP/register.html')

@login_required
def chart_view(request):
    """
    Renders the chatbot UI page with embedded Chart.js container.
    """
    return render(request, 'FSRP/chart.html')
    

def info(request, pk):
    info = Info.objects.get(id=pk)
    room_messages = info.message_set.all()
    participants = info.participants.all()

    if request.method == 'POST':
        message = Message.objects.create(
            user=request.user,
            info=info,
            body=request.POST.get('body')
        )
        info.participants.add(request.user)
        return redirect('info', pk=info.id)

    context = {'info': info, 'room_messages': room_messages,
               'participants': participants}
    return render(request, 'FSRP/info.html', context)

def contact(request):
    return render(request, 'FSRP/contact.html')

@login_required(login_url='login')
def createInfo(request):
    form = InfoForm()
    topics = Topic.objects.all()
    if request.method == 'POST':
        topic_name = request.POST.get('topic')
        topic, created = Topic.objects.get_or_create(name=topic_name)

        Info.objects.create(
            host=request.user,
            topic=topic,
            name=request.POST.get('name'),
            description=request.POST.get('description'),
        )

    context={'form': form, 'topics': topics}
    return render(request, 'FSRP/info_form.html', context)

@login_required(login_url='login')
def updateInfo(request, pk):
    info= Info.objects.get(id=pk)
    form = InfoForm(instance=info)
    topics = Topic.objects.all()
    if request.user != info.host:
        return HttpResponse('Your are not allowed here!!')

    if request.method == 'POST':
        topic_name = request.POST.get('topic')
        topic,created = Topic.objects.get_or_create(name=topic_name)
        info.name = request.POST.get('name')
        info.topic = topic
        info.description = request.POST.get('description')
        info.save()
        return redirect('home')
    context = {'form': form, 'topics': topics, 'info': info}
    return render(request, 'FSRP/info_form.html', context)

@login_required(login_url='login')
def deleteRoom(request, pk):
    info = Info.objects.get(id=pk)

    if request.user != info.host:
        return HttpResponse('Your are not allowed here!!')

    if request.method == 'POST':
        info.delete()
        return redirect('home')
    return render(request, 'FSRP/delete.html', {'obj': info})

@login_required(login_url='login')
def deleteMessage(request, pk):
    message = Message.objects.get(id=pk)

    if request.user != message.user:
        return HttpResponse('Your are not allowed here!!')

    if request.method == 'POST':
        message.delete()
        return redirect('home')
    return render(request, 'FSRP/delete.html', {'obj': message})

def topicsPage(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    topics = Topic.objects.filter(name__icontains=q)
    return render(request, 'FSRP/topics.html', {'topics': topics})

def activityPage(request):
    room_messages = Message.objects.all()
    return render(request, 'FSRP/activity.html', {'room_messages': room_messages})


def index(request):
    infos=Info.objects.all()
    context={'infos': infos}
    return render(request,'index.html',context)

def allInfo(request):
    infos=Info.objects.all()
    context={'infos':infos}
    return render(request, 'FSRP/all_info.html', context)

def userProfile(request, pk):
    try:
        user = User.objects.get(id=pk)
    except User.DoesNotExist:
        messages.success(request, "User does not exist.")

    try:
        user_profile = Profile.objects.get_or_create(user=user)
    except Profile.DoesNotExist:
       messages.success(request, "User does not have a profile.")

    infos = user.info_set.all()
    room_messages = user.message_set.all()
    soil = Soil.objects.get(user = request.user)
    topics = Topic.objects.all()
    context = {
        'user': user, 'infos': infos,
        'room_messages': room_messages,
        'topics': topics, 'user_profile':user_profile, 'soil':soil
    }

    return render(request,'FSRP/user_profile.html', context)

def soil_details(request):
    if request.method == 'POST':
        Soil.objects.create(
        #farm_id= request.POST['farmId']
        user = request.user,
        nitrogen = request.POST['nitrogen_level'],
        potassium = request.POST['potassium_level'],
        phosphorous = request.POST['phosphorous_level'],
        ph = request.POST['ph'],
        rainfall = request.POST['rainfall'],
        )        

    return render(request,'FSRP/recommendation.html')

def update_soil_details(request):
    if request.method == 'POST':
        soil = Soil.objects.get_or_create(user=request.user)
        soil.nitrogen=float(request.POST['nitrogen_level'])
        soil.potassium = float(request.POST['potassium_level'])
        soil.phosphorous = float(request.POST['phosphorous_level'])
        soil.ph = float(request.POST['ph'])
        soil.rainfall= float(request.POST['rainfall'])
        soil.save()
        return redirect('home')
    return render(request,'FSRP/recommendation.html')
def updateUser(request):
    if request.method == 'POST':
        profile = Profile.objects.get_or_create(user=request.user)
        profile.profile_pic = request.FILES['profile_picture']
        profile.name = request.POST['name']
        profile.birth_date = request.POST['birth_date']
        profile.city = request.POST['city']
        profile.state = request.POST['state']
        profile.country = request.POST['country']
        profile.save()
        messages.success(request, 'Your profile has been updated!')
        return redirect('home')
    else:
        profile = Profile.objects.get(user=request.user)
        return render(request, 'FSRP/update-user.html', {'profile': profile})


def recommendation(request):
    context={
        
    }
    return render (request,'FSRP/recommendation.html', context)


def get_weather(request):
    try:
        user = Profile.objects.get(user=request.user)
    except Profile.DoesNotExist:
        messages.success(request, 'no profile')
        return render(request, "FSRP/Weather.html", {"weather_data": None})

    api_key = "7ec0c5ae6f94a2004d09b6880b6f640b"
    longitude = user
    latitude = user.lat
    dt = "2024-01-01"  # Example date, replace with the desired date

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    complete_url = f"{base_url}?appid={api_key}&lat={latitude}&lon={longitude}&units=metric&dt={int(datetime.datetime.strptime(dt, '%Y-%m-%d').timestamp())}"
    response = requests.get(complete_url)

    if response.status_code == 200:
        data = response.json()

        weather_data = data.get("main")
        temperature = weather_data.get("temp")
        pressure = weather_data.get("pressure")
        humidity = weather_data.get("humidity")

        context = {
            "name": "giddy",
            "latitude": latitude,
            "weather_data": {
                "temperature": temperature,
                "pressure": pressure,
                "humidity": humidity,
            }
        }

        return render(request, "FSRP/Weather.html", context)

    else:
        messages.success(request, 'Failed to retrieve weather data')
        return render(request, "FSRP/Weather.html", {"weather_data": None})
    

@login_required(login_url='login')   
def user_location(request):
    ip=requests.get('https://api.ipify.org?format=json')
    ip_data=json.loads(ip.text)
    res = requests.get('http://ip-api.com/json/'+ ip_data["ip"])
    if res.status_code == 200:
        location_data = res.json()
        latitude = location_data.get('lat')
        longitude = location_data.get('lon')
        country= location_data.get('country')
        county = location_data.get('regionName')
        city = location_data.get('city')


        # Save the coordinates to the user's profile
        user, created = Profile.objects.get_or_create(user=request.user)
        user.lat = latitude
        user.lon = longitude
        user.city = city
        user.state = county
        user.country = country
        user.save()
        messages.success(request, 'Location Acquired')
    else:
        messages.error(request,'Location not Acqured')

    return render(request,'FSRP/user_location.html',{'country':country})

def farm_profile(request):

    return render(request,'FSRP/farm_profile.html')
