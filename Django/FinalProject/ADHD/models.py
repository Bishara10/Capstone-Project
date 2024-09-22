from django.db import models
from db_connection import db

# Create your models here.

class Patient(models.Model):
  Subtype = models.CharField(max_length=255)
  Percentage = models.CharField(max_length=255)

signatures_collection = db['signatures']
train_data_collection = db['traindata']