# InfoGAN-pytorch
This is the code for implementation of [InfoGAN](https://arxiv.org/pdf/1606.03657.pdf) written by pytorch.
It is completed on the MNIST dataset.

## Download Dataset
You download the MNIST dataset by this command.
```
python data/downloads.py
```

## Model archtecture
### Generator (G)
This below architecture is for the generator.
There are transposed convolutional networks and linear networks using batch normalization with ReLU activation function.
![Generator](generator_architecture.PNG)

### Discriminator (D)
This below architecture is for the discriminator.
There are convolutional networks and linear networks using batch normalization with leakyReLU activation functions.
![Discriminator](discriminator_architecture.PNG)

## Run the script
After you set the configuration of run script, run the script.
I did set the learning rate of generator and discriminator ten times less than original learning rate.
It was better to train the model.
```
sh run.sh
```

## Experiments
Generator is trained well like this way.   
![Training Steps](all.gif)

As manipulating the latent codes, I got this result.
In this figure, the upper two lines are as manipulating continuous codes and the others are as manipulating categorical codes.
Although it is less distinguishable, we can know that there are some difference when manipulating the latent codes.

![Generated Images by manipulating Latent Code](manipulate100_lr0.1.jpg)


