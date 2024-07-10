from django.db import models

# Create your models here.

class Patient(models.Model):
  Subtype = models.CharField(max_length=255)
  Percentage = models.CharField(max_length=255)
