
from transformers import pipeline
model = pipeline("image-classification", model="microsoft/resnet-50")


"""cuda if availablle"""

response=model("image-classification")
print(response)


