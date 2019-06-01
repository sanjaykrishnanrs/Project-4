# Architectural Basic

## No: of Layers

As a first step, there are two choices - Deep Network and a Shallow Network.

Shallow network should have fat/wide layers to perform good which in turn results in higher number of parameters. As our aim is to have a efficient model which can do it's job with less no: of parameters in a good amount of training time, We can avoid shallow network and try in the path of Deep network.

Second step, would be have enough no: of layers until we have a good 'Receptive Field'

## Receptive Field

Before explaining what is receptive field, I would like to tell the types of receptive field we have in Neural networks. Two types of receptive field :

- Global Receptive field : Defines the viewable area on the input image of the model from a pixel in a layer. So if a pixel in a layer can see 5x5 area in a input image then the Global Receptive field of that layer would be 5x5
- Local Receptive field : Defines the viewable area on the previous layer's output image(which is input of the next layer) from a pixel in next layer. So if layer 2 can see 3x3 on the image delivered by layer 1, then we can say the local receptive field of layer 2 is 3x3

So, we can calculate local as well as global receptive field for each layer in the model considering it's position from the initial layer. For the first layer, Global and Local receptive field would be same.

Final layer before prediction should be in a vantage point w.r.t the input image/object. That means, Global receptive field of the final layer before prediction should be enough to see the interested object in the image. If your model's sole purpose is to learn the object in the image, then this makes sense. If model has to learn background (entire image), then no: of layers should be considered till we have a global receptive field of final layer covers the entire image.

Eg: In our given MNIST digit handwritten dataset, input image is of 28x28, while object stays at almost center all the times with 1pixel margin from sides. So global receptive field of the model (i.e, GRF of final layer before prediction) can be 24x24 or 22x22 also which can serve the purpose of learning the digits.

## Channels and Kernels

- **Kernels**

> 1. Kernels applied on an image give Channels.
> 2. Other names of Kernels are *Feature Extractors*, *Filter*
> 3. Kernel convolves on an image which results in an extracted feature -> channel

- **Channels**

> 1. Feature Maps or Channels are the outcome of Feature extractor convolving on the image
> 2. Number of Channels = Number of Kernels applied on the image
> 3. Each channel carries each features. RGB image has 3 channels where each channel carries the respective color values

## 3x3 Convolution

Convolution is the process of sliding a kernel(matrix) on the input image(matrix) step by step where each step is called stride. We can use a kernel of 'n x n x y ' dimensions for convolution where y will be no: of channels equivalent to the image on which the kernel convolves and n can be 3,5, 7 etc.

We use 3x3 kernels because

> -  ODD squared matrix always uplifts the concept of having an anchor  pixel in centre and symmetrically considering the neghbour pixels. Even  squared matrix won't have anchor pixel and thus resulting in distortion.

> - 3*3` kernel is the matrix which consumes less computational power compared to other matrixes during convolution.` 
>
>   5x5` image --> applying `3x3` kernel --> `3x3` image--applying `3x3` kernel -->`1x1` image
>
>   Here, no: of parameters used = 9 + 9 = 18
>
>   5x5` image --applying `5x5` kernel --> `1x1 image Here, no: of parameters used = 25
>
>   Hence, 3x3 matrix uses less computational power

Due to above advantage of 3x3 kernel, most of the GPUs are tuned/architected to perform well for 3x3 filter.

## Maxpooling

When we do deep networks, it doesn't mean it should go more deep that no: of parameters gets equally big as a shallow/Fully Connected/Dense network.

Also, after detecting edges and gradients, it's really better to let the model amplify/scream out the features it detected and pass to next layers for further detection of textures->patterns->objects. 

Maxpooling does the job for both the above points. It takes a shortcut by cutting down the layers but at the same time not compromising on receptive field. Rather, it doubles up the receptive field and halves the resolution. Maxpooling function finds the prominent feature in the particular area w.r.t pixel intensity and then takes out that value alone and passes to next layer . Thus, it screams out the prominent features to the next layer.

## 1x1 Convolution

1x1 Convolution is special in a lot of ways and also very useful when we need control on the channels of images handles by the network layers.

Usually the no: of layers and maxpooling affects the resolution of the images as it reduces correspondingly the resolution at each step. Then 1x1 Kernels came for having a control on the dimensions. Usually it is used for dimension reductionality (lessening the channels). But, later onwards, we can see for very specific case based models, 1x1 convolutions are used to increase the channels as well. Coming back to dimension reductionality feature, It is usually used adjacent to the Maxpooling as well as before flattening for prediction layer. Both purposes are slightly different.

- Companion of Maxpooling: It can be used as previous step or next step wr.t maxpooling layer. It combines the features detected across different channels and gives it to the next layer. So if 1x1 convolution is used before maxpooling then maxpooling gets benefitted more as it gets more features to amplify. While, in other case where 1x1 is used before after maxpooling, 1x1 gets benefitted from maxpooling where it gets necessary features alone to combine. Anyways, the thumb rule is to use 1x1 convolution with maxpooling after few initial layers only as the model should be given ability to initially detect edges and gradients without interruption before trying to combine or amplify the detected features.

- Companion of Flattening: In some models, we can see 1x1 kernels are used just before Flattening. For MNIST handwritten digit dataset, as we told earlier, we can have model with receptive field 24x24 also. I.e, by the time we reach final layer before prediction there are chances that we will have image resolution of 9x9 or 7x7. And the, we may have to start predicting as convolving a 7x7 image with a 3x3 kernel does not make much sense because only few center pixels will be covered more which won't help in much useful feature detections. In this case, if final predict layer is Softmax, it will take 1D array and Flatten does this. So we add Flatten before Softmax layer. Input for Flattening layer will be like this :

  ```python
  # Input Dimension  = 7x7x32
  # Kernel           = 1x1x32
  # No:of Kernels    = 10
  # Output Dimension = 7x7x10 (Here,10 is the number of channels)
  # Receptive Field  = 18x18
  # NOTE: 1x1 Kernels are used here for dimension(channel) reduction.
  # No effect on output resolution or receptive field
  model.add(Convolution2D(10, 1, activation='relu'))
  
  # Input Dimension  = 7x7x10
  # Kernel           = 7x7x10
  # No:of Kernels    = 10
  # Output Dimension = 1x1x10 (Here,32 is the number of channels)
  # Receptive Field  = 20x20 
  model.add(Convolution2D(10, 7))
  
  # Input Dimension  = 1x1x10
  # Output Dimension = 10
  model.add(Flatten())
  ```
