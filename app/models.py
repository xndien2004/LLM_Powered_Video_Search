from django.db import models
import os

# Create your models here.
class Video(models.Model):
    video_path = models.FileField(upload_to='metadata/videos/') 
    video_name = models.CharField(max_length=100, blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.video_name: 
            last_video = Video.objects.order_by('id').last()
            if last_video:
                last_video_number = int(last_video.video_name)
                self.video_name = f'{last_video_number + 1:04d}'
            else:
                self.video_name = '0001'
        super(Video, self).save(*args, **kwargs)
    
    def delete(self, *args, **kwargs): 
        if self.video_path and os.path.isfile(self.video_path.path):
            os.remove(self.video_path.path)
        super(Video, self).delete(*args, **kwargs)

    def __str__(self):
        return self.video_name

    def get_frames(self):
        return self.frames.all()

    def get_contexts(self):
        return self.contexts.all()


class Frame(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='frames')
    frame_path = models.FileField(upload_to='metadata/frames/')
    frame_name = models.CharField(max_length=100, blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.frame_name: 
            last_frame = Frame.objects.filter(video=self.video).order_by('id').last()
            if last_frame:
                last_frame_number = int(last_frame.frame_name)
                self.frame_name = f'{last_frame_number + 1:04d}'
            else:
                self.frame_name = '0001'
        super(Frame, self).save(*args, **kwargs)

    def __str__(self):
        return self.frame_name


class Context(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='contexts')
    context = models.TextField()

    def __str__(self):
        return self.context[:50]