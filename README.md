# Fetal Ultrasound Grand Challenge: Self-supervised Segmentation of Cervical Ultrasound Images
This work is done as a part of the course EE61008: Medical Image Analysis, a part of the curriculum of Signal Processing and Machine Learning Specialization, Department of Electrical Engineering at Indian Institute of Technology Kharagpur. The problem statement and data is based on the Fetal Ultrasound Grand Challenge from the International Symposium of Biomedical Imaging (ISBI 2025). 

## Challenge Objectives: 
The objective of this project is to develop an algorithm that leverages both labeled and unlabeled data to enhance the accuracy and efficiency of cervical image segmentation in transvaginal ultrasound images. This approach aims to address the challenges posed by limited labeled data in medical imaging by incorporating self-supervised learning techniques.​


## Dataset 
The dataset used in this project is provided by the [Fetal Ultrasound Grand Challenge](https://www.codabench.org/competitions/4781/). It includes a collection of transvaginal ultrasound images, along with corresponding segmentation masks for a subset of the data. The dataset is divided into labeled and unlabeled subsets to facilitate semi-supervised learning.​

## Methodology
The project employs self-supervised learning techniques to improve segmentation performance. The key components of the methodology include:​

- **BYOL (Bootstrap Your Own Latent):** A self-supervised learning approach that learns representations by predicting target network outputs from online network inputs.​

- **Masked Image Modeling:** A technique where parts of the input images are masked, and the model learns to predict the missing parts, encouraging the model to understand the structure and semantics of the images.​

These techniques are integrated into the training pipeline to leverage both labeled and unlabeled data effectively.

## References 

- Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Pires, B., Guo, Z., Azar, M. G., Piot, B., Kavukcuoglu, K., Munos, R., & Valko, M. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*. [[Link](https://arxiv.org/abs/2006.07733)]

- Xie, Z., Zhang, Z., He, X., Lin, Y., & Dai, J. (2022). *SimMIM: A Simple Framework for Masked Image Modeling*. [[Link](https://arxiv.org/abs/2111.09886)]

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. [[Link](https://arxiv.org/abs/2010.11929)]
