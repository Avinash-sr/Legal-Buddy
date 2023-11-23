from django.db import models

# Create your models here.
class laws(models.Model):
    code = models.TextField()
    name = models.TextField()
    description = models.TextField()
    helpline = models.TextField()

    def __str__(self):
        return self.code