# EfficientNet-Lite for Keras and TensorFlow 2
EfficientNet-Lite variants are modified versions of EfficientNet models, better suited for mobile and embedded devices.

The main goal is to mimic the usage of `keras.applications` as well as be fully compatible with Sequential and Functional API, with the ability to fine-tune as other models.

The implementation is based on [official implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite) and [keras implementation](https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py).

All the weights are converted from official [Tensorflow Hub](https://tfhub.dev/).

## Available Models
- Model
  * EfficientNetLiteB0
  * EfficientNetLiteB1
  * EfficientNetLiteB2
  * EfficientNetLiteB3
  * EfficientNetLiteB4

## Requirements
- Python 3+ (Recommended: 3.13.0+)
- tensorflow 2.2+ (Recommended: 2.20.0+)

## Usage 
Include the `efficientnet_lite.py` and the folder `weights` in your file directory of your project. The path of `efficientnet_lite.py` and `weights` must be the same.

The usage is the same as using native models of `keras.applications`.
```python
# Import package
from efficientnet_lite import EfficientNetLiteB0, EfficientNetLiteB1, EfficientNetLiteB2, EfficientNetLiteB3, EfficientNetLiteB4

# weights = None|"imagenet"|path_to_your_weights_file to load None or pre-trained imageNet1K weights or custom weights. (Default: "imagenet")
# include_top = True|False to include/exclude the final classification layer. (Default: True)
# You can pass model_name to set the model's name. (Default: "efficientnet_lite_b{0-4}")
# See the efficinetnet_lite.py file for more details.
cnn_model = EfficientNetLiteB0(input_shape = (224, 224, 3), weights = "imagenet", include_top = False)

# You can freeze or unfreeze the whole model or specific layers as usual for fine-tuning.
cnn_model.trainable = True

print(f"Name : {cnn_model.name}")
print(f"Total Layers : {len(cnn_model.layers)}")
cnn_model.summary(show_trainable=True)

# Example Model Building with Top Classification Layer
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

imput_tensor = cnn_model.input
x = GlobalAveragePooling2D()(cnn_model.output)
output_tensor = Dense(10, activation='softmax')(x)

model = Model(inputs=imput_tensor, outputs=output_tensor, name="EfficientNetLiteB0_CustomTop")
model.summary(show_trainable=True)

model.compile(...)
model.fit(...)
```

### Input shapes
The following table shows the default input shapes for each model variant.

| Model variant | Input shape |
|:-------------:|:-----------:|
|       B0      | `224,224`   |
|       B1      | `240,240`   |
|       B2      | `260,260`   |
|       B3      | `280,280`   |
|       B4      | `300,300`   |

### Preprocessing
The models expect image values in the range `[0, 1]` of type `float32`.
You can normalize your image by `image / 255.0` before passing your image into the model.

For safety, you can use `preprocess_input(image)` available inside the `efficientnet_lite.py` on your image beforehand.

## Kaggle Adaptation
If you are working on Kaggle, then you can use these [Kaggle models](https://www.kaggle.com/models/samunislamsamun/efficientnet-lite-models-with-imagenet1k-weights/) or [weights-only](https://www.kaggle.com/models/samunislamsamun/efficientnet-lite-imagenet-model-weights/).
Moreover, you can see this [Kaggle Notebook](https://www.kaggle.com/code/samunislamsamun/efficientnet-lite-for-keras-and-tensorflow-2) for more details.

Alternatively, you can install this repo in a Kaggle Notebook like below.
```python
!git clone https://github.com/Samun-Islam-49/EfficientNet-Lite-Keras-TensorFlow2.git
```
```python
import sys
sys.path.append("/kaggle/working/EfficientNet-Lite-Keras-TensorFlow2")
```
Then you can use it like shown above.

## Last Words
Feel free to try this repo, share your opinions or comments, and contribute. If this helps, consider giving it a star.

## Contributors
- Samun Islam (samunislam49@gmail.com)
