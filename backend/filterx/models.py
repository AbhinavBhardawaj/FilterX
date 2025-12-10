"""Models for the FilterX app."""
from django.db import models


class FilteredContent(models.Model):
    """Model to store filtered content metadata."""
    
    url = models.URLField(max_length=500)
    is_nsfw = models.BooleanField(default=False)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.url} - NSFW: {self.is_nsfw}"
