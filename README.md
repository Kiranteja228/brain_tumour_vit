# brain_tumour_vit
MRI Brain Tumour Detection using Vision Transformers

This repo contains the Notebook file of training, testing and saving a fine-tuned vision transformer model for detecting brain tumours from MRI.

# Quick Inference

## To run this model using Pipeline:

from transformers import pipeline
pipe = pipeline("image-classification", model="kiranteja/mri_brain_tumour_vision_transformers")

## To load the model directly

from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("kiranteja/mri_brain_tumour_vision_transformers")
model = AutoModelForImageClassification.from_pretrained("kiranteja/mri_brain_tumour_vision_transformers")
