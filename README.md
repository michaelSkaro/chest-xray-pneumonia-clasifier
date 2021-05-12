# Pneumonia Disease Classification from Patient X-Ray Images

1.3 million people are diagnosed with Pneumonia as an upper respiratory disease stemming from either a bacterial or viral infection each year in the United States. The the canonical diagnostic modus operandi is through professional review of chest X-ray images. The volume of image data and medically standardized data production pipeline represents a clear opportunity to develop diagnostic aids for medical professionals using computer vision and deep learning architectures. We collected data from the publicly available Kaggle X-ray image data repository to train a convolutional neural network to distinguish between healthy and infected chest X-ray images. Our model achieved accuracy: 0.83, F1-score: 0.83, precision: 0.79 and recall: 0.99 on the test set using a standard transfer learning process with image augmentation.

The [binary classifier](./binary_classification.ipynb) is the main file containing our overall workflow. It starts with the description of the project and an exploratory data analysis, and proceeds to neural networks that classify X-ray images of healthy and pneumonia patients. The PyTorch model checkpoint is available [here](./model_weights/aug_model.pt.ckpt).

The [multiclassifier](./multiclass_classification.ipynb) extends the binary classifier to three classes: normal, bacterial pneumonia and viral pneumonia. Model weights are [also available](./model_weights/multiclass_model.pt.ckpt).

The [model structure](./core/model.py) and [data loader](./core/data.py) scripts are defined in separate files so that they could be reused in the two notebooks above.

Finally, the feature extractor models and an exploration of their concensus are shown in [this notebook](./feature_extractor.ipynb). Some further analyses of the sample images are in the [model interpretability notebook](./Model_interpretability_Chest_x_ray.ipynb).
