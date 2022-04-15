> 📋 A README.md for code accompanying a Pattern recognition paper

# Does Object Recognition Work for Everyone?

This repository is the official implementation of paper Does Object Recognition Work for Everyone?. The implementation
of the algorithm developed by our own group is contained in `custom-object-detector-implemented-by-own.py` file. Since
training based on full training dataset might be time-consuming, we decided to also include a pre-trained model with our
repository so that it will be easier for the respectful evaluators to re-run and evaluate our algorithm. We used the
tensorflow builtin library files to run and train the model. For testing purposes we also included annotated bounding
boxes, related xml files and also `training.record` and `testing.record` to later fed them in training phase

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Datasets for training are included in repository. The images and all related annotations are included in `images/training` folder

## Training

Training with current configuration can take upto 6 hours to train the model(s) in the paper, run this command:

```train
python model_main_tf2.py --pipeline_config_path=training/pipeline.config --model_dir=training --alsologtostderr -W ignore
```
> Once you trained the model, you can then export the checkpoints and all model-related files using the command below:

```export
 python research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir training \
    --output_directory inference_graph \
    --pipeline_config_path training/pipeline.config
```

> 📋 To train the model first we have to clone the tensorflow model repo: `git clone https://github.com/tensorflow/models.git`, Then we have to navigate to the `models/research` path and compile the object detection scripts using the following command `protoc object_detection/protos/*.proto --python_out=.` We have to copy the compiled setup scripts using this command `cp object_detection/packages/tf2/setup.py .` then we will be able to install object detection framework using this command: `python -m pip install .`

> Next we will be able to locate the `model_main_tf2.py` inside `research/object-detection` folder which will be used for training the model

> After training the model you have to run the following command to export the saved model `python research/object_detection/exporter_main_v2.py \
--trained_checkpoint_dir training \
--output_directory inference_graph \
--pipeline_config_path training/pipeline.config`

> We will be able to fed the saved model to our evaluation script in next step

## Evaluation

To evaluate my model on Dollar Street dataset, run the following command:

```eval
python object_Detection.py
```

> 📋 We used different hyperparameters, fine-tuned in order to obtain the highest accuracy in detection, you can set the hyperparameter values inside pipeline.config file. You can find the pipeline configuration inside `training/pipeline.config` folder. We used `training_steps`, `learning_rate`, and `batch_size`. We found that we can obtain the maximum value of average accuracy for all three classes which is 86%. the values of hyperparameters are listed accordingly `num_steps: 50000` , `learning_rate_base=0.8`, `batch_size=300`

## Results

Our model achieves the following performance on our evaluation phase splitting training dataset with 80/20 ratio (80% of
the data used for training and 20% used for testing):

### [Object Detection using Mobilenet V2](https://paperswithcode.com/model/mobilenet-v2)

| Model name         | Accuracy (Average value for all three classes)  | 
| ------------------ |---------------- | 
| Mobilenet V2   |     86%         |


## Contributing

> [![License: CC0-1.0](https://licensebuttons.net/l/zero/1.0/80x15.png)](http://creativecommons.org/publicdomain/zero/1.0/)
