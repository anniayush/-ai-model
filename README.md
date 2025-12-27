Fine-tuning builds on transfer learning: a model pre-trained on a large dataset (e.g., ImageNet or a big medical corpus) is adapted to a more specific cancer detection task. In practice, earlier layers (that learn generic edges, textures, shapes) are often frozen, while later layers and newly added classification heads are retrained on the cancer dataset.
​
<br>


For image-based cancer detection (e.g., histopathology or dermoscopy images), CNNs like ResNet, VGG, DenseNet or custom CNNs are common starting points.


<br>​

For report or note understanding (e.g., pathology reports), large language models can be fine-tuned with medical text and labels for cancer staging, site, or treatment plans.
