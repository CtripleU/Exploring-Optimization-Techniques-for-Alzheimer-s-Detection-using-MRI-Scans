# Alzheimer's MRI Classification Project

## 1. Project Description

This project aims to develop machine learning models to classify Alzheimer's disease severity based on MRI images. The models are trained to predict the severity level of Alzheimer's disease based on MRI images preprocessed to a size of 128x128 pixels. The core focus lies on implementing and evaluating an optimized Convolutional Neural Network (CNN) model to achieve superior performance compared to a baseline model.

## 2. Dataset and Preprocessing

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset). It consists of preprocessed MRI images collected from various sources, including hospitals and public repositories. It contains images of four classes representing different levels of Alzheimer's disease severity:

- Class 1: Mild Demented (896 images)
- Class 2: Moderate Demented (64 images)
- Class 3: Non Demented (3200 images)
- Class 4: Very Mild Demented (2240 images)

Each image in the dataset is resized to 128x128 pixels and normalized to pixel values between 0 and 1. 
The primary feature is the MRI image itself, which captures the structural information of the brain. Since the images are preprocessed, intensity values likely represent tissue characteristics potentially indicative of Alzheimer's Disease progression. These features directly contribute to model training and prediction by providing detailed information about brain structure and abnormalities associated with Alzheimer's disease.

## 3. Model Architectures
### 3.1 Simple Model (Baseline)
A basic CNN model serves as a reference for comparison with the optimized model. It lacks explicit optimization techniques to highlight the impact of the optimizations in the advanced model.

### 3.2 Optimized Model
This model incorporates several optimization techniques to address overfitting and improve generalization:

#### A. L2 Regularization:
- **Explanation**: L2 regularization, also known as weight decay, adds a     penalty term to the loss function, forcing the model's weights to stay small during training. This helps prevent overfitting by discouraging complex weight values.
- **Relevance to Projec**t: In this project, L2 regularization is applied to convolutional layers using the kernel_regularizer parameter. By regularizing the weights of convolutional kernels, the model learns simpler and more generalized features from the input images, which can improve its performance on unseen data.
- **Parameter and Tuning**: The regularization strength (0.001 in this case) is a hyperparameter that controls the impact of the regularization term on the loss function. This value is selected through experimentation and validation on a separate validation set. A small value is chosen to avoid overly aggressive regularization, which could lead to underfitting.


### B. Dropout Regularization

- **Explanation**: Dropout regularization is used to prevent overfitting by randomly dropping neurons during training. This technique helps the model learn more robust features and reduces reliance on specific neurons.
- **Relevance to Project**: Dropout is applied to the fully connected layers of the model using the `Dropout` layer. By randomly dropping neurons, the model becomes less sensitive to specific features and learns a more diverse set of representations, which can lead to better performance on unseen data.
- **Parameter and Tuning**: The dropout rate (0.25 in this case) is a hyperparameter that controls the fraction of neurons to drop during training. This value is selected through experimentation and validation on a separate validation set. A moderate dropout rate is chosen to prevent overfitting without significantly impacting the model's learning capacity.

### C. Batch Normalization:

- **Explanation**: Batch Normalization is a technique that normalizes the activations of each layer across mini-batches during training. It helps stabilize and speed up the training process by reducing internal covariate shift and allowing higher learning rates.
- **Relevance to Project**: Batch Normalization is applied after the convolutional layers using the BatchNormalization layer. By normalizing the activations, Batch Normalization helps the model converge faster and reduces the dependence on the initialization scheme, leading to more stable training and better performance.
- **Parameter and Tuning**: Batch Normalization does not introduce additional hyperparameters to tune. The default settings provided by TensorFlow's BatchNormalization layer are used in this project.

### D. Early Stopping

- **Explanation**: Early stopping is utilized to prevent overfitting by monitoring the validation loss during training. If the validation loss does not improve for a certain number of epochs, training is stopped to avoid further overfitting.
- **Relevance to Project**: Early stopping is implemented using the EarlyStopping callback in TensorFlow's Keras API. By monitoring the validation loss, the training process is terminated early if there is no improvement in validation performance for a certain number of epochs (patience). This prevents the model from overfitting to the training data and ensures better generalization to unseen data.
- **Parameter and Tuning**: The patience parameter (set to 5 in this case) determines the number of epochs to wait before stopping training when no improvement is observed in the validation loss. This value is selected based on the trade-off between early stopping and model convergence. A higher patience value allows the model to train for longer, potentially improving performance, while a lower value stops training earlier to prevent overfitting.

### E. Adam with Learning Rate Decay:

- **Explanation**: Adam (Adaptive Moment Estimation) is an adaptive optimization algorithm that combines the benefits of both AdaGrad and RMSProp. It dynamically adjusts the learning rate for each parameter based on the first and second moments of the gradients, allowing for faster convergence and better performance.
- **Relevance to Project**: Adam optimizer is used to update the model's weights during training, optimizing the loss function with respect to the model parameters. Additionally, a learning rate decay schedule is applied to gradually reduce the learning rate over time, which can help fine-tune the model and improve its generalization performance.
- **Parameter and Tuning**:  The default settings provided by TensorFlow's Adam optimizer are used in this project, including the initial learning rate and decay schedule. These parameters are chosen based on empirical evidence and best practices in deep learning.

## 4. Training and Evaluation

- **Training Setup**:  Both models are trained with a fixed number of epochs (10) and batch size (32) to ensure a controlled comparison.
- **Early Stopping**: The optimized model employs early stopping with a patience of 5 epochs to prevent overfitting. Training ceases if the validation loss fails to improve for 5 consecutive epochs.
- **Evaluation Metrics**:  Models are evaluated on a held-out test set using comprehensive metrics including accuracy, precision, recall, F1-score, specificity, and sensitivity. These metrics provide a holistic understanding of model performance across different aspects.
- **Confusion Matrices**: Confusion matrices are generated to visualize the distribution of correct and incorrect predictions for each AD severity class. These matrices offer valuable insights into potential class imbalances and areas for improvement.

## 5. Expected Results and Discussion

The optimized model, equipped with regularization techniques and dropout, is expected to demonstrate superior performance compared to the simple model. Here's a breakdown of the anticipated benefits:

- **Reduced Overfitting**: L2 regularization and dropout should prevent the model from memorizing training data and enhance generalization to unseen examples.
- **Improved Training Stability**: Batch normalization should stabilize the training process, allowing the model to converge faster and potentially achieve better solutions.
- **Deeper Analysis**: Evaluation metrics and confusion matrices will provide a detailed understanding of the model's strengths and weaknesses for each Alzheimer's Disease severity class. This analysis can guide further model refinements.

## 6. Conclusion

This project showcases the power of machine learning for AD classification using MRI scans. By incorporating optimization techniques, the model achieves a potential improvement in accuracy and robustness compared to a basic model. Further exploration could involve:

- **Hyperparameter Tuning**: Optimizing hyperparameters like learning rate, dropout rate, and L2 regularization strength for potentially even better performance.
- **Network Architectures**: Experimenting with different CNN architectures, such as deeper networks with residual connections, to explore their impact on classification accuracy.
- **Transfer Learning**: Leveraging pre-trained models on larger medical image datasets to potentially improve model performance and reduce training time.

This project contributes to the ongoing research on using machine learning for early and non-invasive diagnosis of Alzheimer's Disease.

## 7. Running the Saved Models
The project includes two trained models saved as .pkl files:

- `saved_models/simple_model.pkl`: The baseline model without optimizations.
- `saved_models/optimized_model.pkl`: The model incorporating L2 regularization, dropout, and Batch Normalization.

To run these models, you'll need the following:

- Python 3.6 or later
- TensorFlow (tested with TensorFlow 2.x)
- NumPy
- scikit-learn (optional, for label encoding)
