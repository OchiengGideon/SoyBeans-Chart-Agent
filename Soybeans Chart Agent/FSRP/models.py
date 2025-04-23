from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver


# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_pic = models.ImageField(default='default.jpg', upload_to='profile_pics')
    date_of_birth = models.DateField(blank=True, null=True)
    phone_number = models.CharField(max_length=12, blank=True)
    address = models.CharField(max_length=100, blank=True)
    lat = models.FloatField(blank=True, null=True)
    lon = models.FloatField(blank=True, null=True)
    city = models.CharField(max_length=50, blank=True)
    state = models.CharField(max_length=50, blank=True)
    country = models.CharField(max_length=50, blank=True)
    postal_code = models.CharField(max_length=10, blank=True)

    @receiver(post_save, sender=User)
    def create_profile(sender, instance, created, **kwargs):
       if created:
        Profile.objects.create(user=instance)
        
    @receiver(post_save, sender=User)
    def save_profile(sender, instance, **kwargs):
        instance.profile.save()
           
    def __str__(self):
        return f'{self.user.username} Profile'

class Topic(models.Model):
  name= models.CharField(max_length=200)

  def __str__(self):
    return self.name
class Info(models.Model):
  host = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
  topic = models.ForeignKey(Topic, on_delete=models.SET_NULL, null=True)
  name = models.CharField(max_length=200)
  description = models.TextField(null=True, blank=True)
  participants = models.ManyToManyField(
      User, related_name='participants', blank=True)
  updated = models.DateTimeField(auto_now=True)
  created = models.DateTimeField(auto_now_add=True)

  class Meta:
        ordering = ['-updated', '-created']

  def __str__(self):
        return self.name
  
class Message(models.Model):
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  info = models.ForeignKey(Info, on_delete=models.CASCADE)
  body = models.TextField()
  updated = models.DateTimeField(auto_now=True)
  created =models.DateTimeField(auto_now_add=True)

  def __str__(self):
    return self.body[0:50] 


class Soil(models.Model):
   user = models.OneToOneField(User, on_delete=models.CASCADE)
   nitrogen = models.FloatField(blank=True, null=True)
   phosphorous = models.FloatField(blank=True, null=True)
   potassium = models.FloatField(blank=True, null=True)
   rainfall = models.FloatField(blank=True, null=True)
   ph = models.FloatField(blank=True, null=True)

   @receiver(post_save, sender=User)
   def create_profile(sender, instance, created, **kwargs):
       if created:
        Soil.objects.create(user=instance)
        
   @receiver(post_save, sender=User)
   def save_profile(sender, instance, **kwargs):
        instance.soil.save()

   def __str__(self):
        return f"{self.user.username}'s Soil Profile"

class FarmerQuery(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name="farmer_queries")
    query_text    = models.TextField()
    response_text = models.TextField()
    timestamp     = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"{self.user}@{self.timestamp}"

class Reminder(models.Model):
    user       = models.ForeignKey(User, on_delete=models.CASCADE)
    message    = models.TextField()
    send_at    = models.DateTimeField()
    sent       = models.BooleanField(default=False)

    def __str__(self):
        return f"Reminder for {self.user} at {self.send_at}"