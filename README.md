![DRIVER_DROWSINESS_DETECTION_(CNN)-OneAPI](https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/3383df7e-107e-459c-81be-4d290c8c1ee3.png)
    ![made-with-jupyter-notebook](https://user-images.githubusercontent.com/130077430/230479936-93dbcbd0-275b-4af7-9231-cceeb91d8a84.svg)          ![migrated-to-oneapi](https://user-images.githubusercontent.com/130077430/230487901-cbcdf13f-1d36-477d-9a7c-1917fa579da9.svg)![built-by-team-geeks](https://user-images.githubusercontent.com/130077430/230486285-e9e8fdbc-4579-4d0e-a448-550b423199b2.svg)
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/gangeshbaskerr/DriverDrowsinessDetection-OneAPI"> <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/gangeshbaskerr/DriverDrowsinessDetection-OneAPI?style=social">
<hr/>

# Inspiration  <img src="https://user-images.githubusercontent.com/130077430/230579469-c1263cef-784e-4845-93fb-2f73544e49e1.png" width="90" height="80"> 
Throughout our lives, we've been exposed to countless reports and stories in newspapers and online platforms, underscoring the distressing prevalence of sexual harassment incidents. What strikes me even more profoundly is the significant reluctance victims often experience when reporting such incidents due to fears of retaliation or social stigma. This reporting hesitancy and the lack of practical technological solutions have intensified my resolve to address this critical issue. The absence of practical tools and measures to combat such behaviour compelled me to develop the "Sexual Harassment Detection at Workplace" project. The driving force behind this endeavour is not only to close the gap between the gravity of the problem and the available solutions but also to contribute to a safer and more inclusive work environment by harnessing the potential of innovative technological advancements.

<img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/c3cc1325-ccd7-43cb-8a01-98a4882a6d14" width="330" height="280"><img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/8a110924-77f3-4e0c-9155-556afdde1105.png" width="330" height="280"><img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/4249f60e-e49c-4711-8b1e-c9911466fd84" width="330" height="280">

<hr/>

# Problem Statement <img src="https://user-images.githubusercontent.com/130077430/230730194-a7389fed-f5fd-48d3-856a-0212057f2500.png" width="90" height="80">

My main goal is to build an innovative technological solution that could detect sexual harassment in real-time, which is much needed to fill in the gap left void by traditional methods,  by which I aim to create a safer work environment, overcome reporting hesitancy, support HR and legal management, reduce psychological and physical stress and do early detection and prevention.

<hr/>

# Introduction <img src="https://user-images.githubusercontent.com/72274851/152814876-73362bcc-bde6-411f-ba80-235e911f276f.gif" width="90" height="90">

Recognizing and tackling harassment instances can be complex due to victim hesitation prompted by fear or stigma. We aim to develop a CNN-based model to detect workplace harassment, differentiating it accurately from normal behavior's in videos. Our solution leverages a CNN-based model to effectively detect harassment in videos, contributing to a safer workplace and fostering a culture of inclusivity.
 

<hr/>

# Approach <img src="https://cdn0.iconfinder.com/data/icons/data-science-2-1/66/119-512.png" width="90" height="80"> 

Our model uses a CNN(Convolutional Neural Network) for detection since the dataset uses images using a Deep Learning model to make better detection. A dataset containing images will be used, out of which half will have harassment images, and the other half will show a normal work environment. We will follow an 80:20 split, where 80% of the images will be used for training our model, and the remaining 20% will be reserved for testing the model. To further enhance the accuracy of the detection system, we have 
implemented Transfer Learning techniques by leveraging pre-trained models such as VGG16, Xception, and others. These models have been trained on large-scale datasets and come with high accuracy levels.

<hr/>

# Procedure <img src="https://th.bing.com/th/id/R.02832177b40b49d50674126476f980c3?rik=aXibwvpQe645bg&riu=http%3a%2f%2fwww.clipartbest.com%2fcliparts%2fjcx%2f6rb%2fjcx6rbngi.png&ehk=xllVkMLnEE%2fEXx%2fnWbpceiVVfvTNGJmODcZ9fEBJVGA%3d&risl=&pid=ImgRaw&r=0" width="120" height="100"> 

## 1️⃣ Pre-install all the required libraries
       1) OpenCV
       2) Keras
       3) Numpy
       4) Pandas
       5) OS
## 2️⃣ Understand the dataset
       As no solution was available to this problem before, this project had no existing dataset online.
       So, we had to create the dataset ourselves, which was a tedious process.
       it has 2 folder which are :
       1) yes - having 1000 pictures(Harassment)
       2) no - having 1000 pictures(Healthy work environment)
 <img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/4791552d-9924-42e7-b710-4bd717e860a6.jpg" width="495" height="500"> <img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/bf8a8672-cf58-46a6-a466-b72fe0be1bac.jpg" width="495" height="500">
 
## 3️⃣ Data preprocessing
1. preprocess the images from the yes and no folder.

2. Resizing all the images to the same dimension (224,224,3)
  
3. Convert the images into numpy arrays.
   
4. The dataset will be split into training, validation, and testing sets.

## 4️⃣ Build and train the VGG16 model
The VGG16 model is designed and trained to classify images as either Harassment or Healthy.

<img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/1baa15d7-8848-4e6d-84d2-e2888943d2c1.png" width="650" height="500">  <img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/17fa9906-08f0-48cf-a20a-7d811252215a.png" width="340" height="500">

## 5️⃣ Train the model using Intel OneAPI to get better results

<img src="https://user-images.githubusercontent.com/72274851/218504609-585bcebe-5101-4477-bdd2-3a1ba13a64a8.png" width="190" height="100"><img src="https://aditech.in/wp-content/uploads/2020/07/image_2020_07_17T06_08_48_297Z.png" width="800" height="100">

**How does OneApi provide better performance :**
    
Today’s computer systems are heterogeneous and include CPUs, GPUs, FPGAs, and other accelerators. The different architectures exhibit varied                   characteristics that can be matched to specific workloads for the best performance.
Having multiple types of compute architectures leads to different programming and optimization needs. oneAPI and SYCL provide a programming model, whether through direct programming or libraries, that can be utilized to develop software tailored to each of the architectures.
    
**Advantages of using OneAPI :**

1) We can use Single code for both CPU and GPU (heterogeneous computing)
2) To implement machine learning based IoT projects easily with less hardwares as the machine learning part happens in cloud
3) To process files faster ie. it takes less time to run the epochs
4) OneAPI allows users to transcend Hardware restrictions and provide better performance for low powered computers
5) Accuracy will improve while using OneAPI

<img src="https://user-images.githubusercontent.com/130077430/230733185-94fbda70-6fe6-40af-985c-d7f8a74a3521.jpg" width="495" height="400"><img src="https://user-images.githubusercontent.com/130077430/230733189-78e03097-7c88-4f42-9c0e-159e58aa7972.jpg" width="495" height="400">

To migrate your project to OneAPI : 
[click here!](https://devcloud.intel.com/oneapi/get_started/) to get started

For reference : [click here!](https://www.youtube.com/watch?v=NkJXCalgmeU)
    
    
## 6️⃣ Save the model
       save the model to calculate the accuracy and loss
    
<hr/>

# Accuracy and Loss      <img src="https://user-images.githubusercontent.com/130077430/230577475-9af43d03-1a50-41c2-99b2-e1a28b69c84e.png" width="90" height="80">

We did 80 epochs, to get a good accuracy from the model i.e. 98% for training accuracy and 97% for validation accuracy.

<img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/7dde241e-e9cd-4a49-8977-dcae1e12da3c.png" width="1000" height="500"> 

<hr/>

# Output <img src="https://cdn4.iconfinder.com/data/icons/business-startup-36/64/552-512.png" width="90" height="80">
## Harassment

<p align="middle"><img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/09892e38-2275-402a-b8be-5df1173076c3.gif" width="500" height="500">
    
## No Harassment

<p align="middle"><img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/15c0173b-bf4c-445c-9ce3-89cb00eb2e51.gif" width="500" height="500">
    
<hr/>

# Learnings <img src="https://user-images.githubusercontent.com/130077430/230583675-33ad7480-857b-451f-a64b-3c45f21d390a.png" width="90" height="80">

<img src="https://user-images.githubusercontent.com/72274851/218504609-585bcebe-5101-4477-bdd2-3a1ba13a64a8.png" width="190" height="100"><img src="https://user-images.githubusercontent.com/72274851/220130227-3c48e87b-3e68-4f1c-b0e4-8e3ad9a4805a.png" width="800" height="100">

1) **Building a CNN model using Intel oneDNN :**
    OneAPI is an open-source software toolkit developed by Intel that simplifies the development of high-performance, heterogeneous applications. It allows       developers to write code that can run efficiently on a variety of architectures, including CPUs, GPUs, and FPGAs. oneDNN (Deep Neural Network) is a part     of oneAPI and is an optimized library for deep learning. It provides highly optimized building blocks for neural network models that run efficiently on a variety of hardware platforms.
   
2) **Machine Learning :**
    _How to use machine learning for identifiying the facial features from a drivers face to detect drowsiness._
   
3) **Convolutional Neural Network(CNN) :**
    _How to build, train and fine-tune convolutional neural networks for image and video classification._
   
4) **Preprocessinig the datasets :**
    _How to preprocess the data dowloaded from kaggle so that the machine learning could happen in a much better and efficient way._

    
5) **Team work :**
    _Collaborating and communicating effectively in a team to deliver a project._
   
6) **Understanding the need for a sexual harassment detection system**
_These are just a few examples of the knowledge and skills that i likely gained while building this project. Overall, building a sexual harasssment detection model  is a challenging and rewarding experience that requires a combination of technical expertise and knowledge ._

<hr/>

# Project Deployment <img src="https://user-images.githubusercontent.com/130077430/230725195-2f024fca-9cae-4e91-85dc-4c12e0e1fcb0.png" width="90" height="80">

We have built an app using Flutter. Flutter helps Build, test, and deploy beautiful mobile, web, desktop, and embedded apps from a single codebase. It is a cross-platform app development framework by Google which goes hand in hand with the model to help ensure the safety of the user and other commuters. 

As soon as the model detects drowsiness, the model will send an API request call to the client app, which notifies the user to take some rest and shows the navigation option to the nearest resting places. If the user isn't drowsy, the app will give 10 seconds buffer time within which the user can confirm that he isn't sleepy by pressing the prompt on the screen. If the user is drowsy he will get a option for getting driving assistance from the nearby driving service providers. If the user has been detected drowsy more than three times within 10 minutes, a notification is sent to the highway patrol and the nearby drivers as a concern for the safety of other drivers and the drowsy driver.

<img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/4b16dac3-96a4-4f90-8694-f67d30b6268c" width="247" height="500"><img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/39462df2-25da-4ec0-b430-d71747678136" width="247" height="500"><img src="https://github.com/gangeshbaskerr/Sexual-Harassment-Detection/assets/130077430/f4598f49-c779-4a21-8dd1-4c5e2808f6f1" width="497" height="500">
<hr/>

# One more thing <img src="https://cdn.freebiesupply.com/logos/large/2x/apple1-logo-png-transparent.png" width="60" height="60">

<p align="middle"><img src="https://th.bing.com/th/id/R.cfabfe3a83a918b326ede9efb1d7ee8b?rik=sxInqysclnUS1A&riu=http%3a%2f%2fmedia.idownloadblog.com%2fwp-content%2fuploads%2f2015%2f08%2fSteve-Jobs-One-More-Thing.jpg&ehk=VbXo3DNGszgubtTtwYXhvwQyxwDKVJ%2bW7%2b0%2bproDQ%2fM%3d&risl=&pid=ImgRaw&r=0" width="800" height="300">
    
1) **EARLY WARNING SYSTEM :**

    Through behavioural analysis it predicts non-desirable behavioural patterns and warns potential threats.

   
2) **EMOTION AND INTENT ANALYSIS :**
   
    Analyzes emotions and intentions in conversations, identifying subtle signs of manipulation, gaslighting, and emotional abuse.
   
3) **CULTURAL NORMALISATION :**

   Adapt the model to different cultural contexts for multinational organizations.

4) **REALTIME ADAPTIVE LEARNING :**

   Periodically re-train the model with new data to adapt to evolving and emerging harassment tactics.

5) **ENHANCING SOCIETAL IMPACT :**

   Integrating advanced predictive machine learning models with a focus on addressing and resolving societal challenges.

6) **CONTEXTUAL DOMAIN EXPANSION AND ADAPTION :**

   Expanding a model's capabilities to detect sexual harassment in various places and environments.

   
<hr/>

# Conclusion <img src="https://user-images.githubusercontent.com/130077430/230730394-3dfbc977-435b-4a6f-bfa3-fc193606f0e0.png" width="90" height="80">

Through this project we can create a safer work environment, Support HR and Legal management, Reduce psychological and physical stress, early detection and prevention and overcome reporting hesitancy. If this project is used efficiently it may also lead to huge decrease in percentage of sexual harassment cases .

To view this project on Intel DevMesh : [click here!](https://devmesh.intel.com/projects/driver-drowsiness-detection-70c5e4)
<hr/>
