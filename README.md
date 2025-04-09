# Neural Network Visualization: Interactive Decision Boundary Analysis

![TensorFlow.js Version](https://img.shields.io/badge/TensorFlow.js-3.x-orange.svg)
![React Version](https://img.shields.io/badge/React-18.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Abstract

This repository presents an interactive web-based framework for visualizing neural network decision boundaries and learning dynamics in real-time. The implementation leverages TensorFlow.js for in-browser model training and inference, while React provides a responsive user interface for parameter manipulation and visualization rendering. The framework addresses several limitations in existing browser-based deep learning visualization tools, particularly focusing on memory management challenges inherent to JavaScript-based tensor operations.

## Core Features

### 1. Interactive Neural Network Training

- **Real-time Parameter Adjustment**: Dynamically alter learning rate, activation functions, layer count, and neuron density
- **Multi-dataset Support**: Visualize learning on XOR, concentric circles, and regression tasks
- **Epoch-by-epoch Visualization**: Monitor decision boundary evolution during training

### 2. Advanced Tensor Management

- **Multi-tier Memory Handling**: Progressive fallback approaches for reliable tensor operations in memory-constrained environments
- **Layer-wise Prediction**: Custom implementation of forward propagation with explicit shape reconciliation
- **Engine-scope Isolation**: Deterministic tensor disposal to prevent memory leaks

### 3. Visualization Components

- **Decision Boundary Rendering**: High-resolution gradient visualization of model predictions
- **Training Metrics**: Real-time loss and accuracy plots
- **Dataset Visualization**: Interactive data point display with class identification

## Theoretical Background

The visualization framework demonstrates several key concepts in neural network learning:

- **Decision Boundary Formation**: Visual representation of how hidden layers transform the input space
- **Gradient Descent Dynamics**: Observation of optimization trajectory through loss landscapes
- **Activation Function Influence**: Comparative analysis of activation function impact on decision boundary smoothness
- **Overfitting Visualization**: Identification of model complexity's relationship to generalization capability

## Installation

```bash
# Clone the repository
git clone https://github.com/umerkhan95/Neural-Network-visualisation.git
cd Neural-Network-visualisation

# Install dependencies
npm install

# Start the development server
npm start
```

The application will be available at [http://localhost:3000](http://localhost:3000).

## Implementation Details

### Memory Management Strategy

The system implements a novel multi-tier approach to TensorFlow.js memory management:

1. **Primary Approach**: Standard batch prediction with managed memory
2. **Secondary Approach**: Direct layer application with tensor shape validation
3. **Fallback Visualization**: Gradient-based rendering with simplified decision boundaries

This progressive strategy ensures visual continuity even when facing TensorFlow.js's internal exceptions like "disposeNewTensors is not defined" and "ENGINE is not defined".

### Training Methodology

Training procedures are implemented with explicit control over tensor lifecycle:

```typescript
// Example: Scope-based tensor management during training
tf.engine().startScope();
try {
  // Perform tensor operations with explicit cleanup
  const trainingInputs = xorData.inputs.clone();
  const trainingOutputs = xorData.outputs.clone();
  
  // Model training with custom callbacks
  await model.fit(trainingInputs, trainingOutputs, {
    batchSize,
    epochs: remainingEpochs,
    callbacks
  });
} finally {
  // Guarantee tensor cleanup
  tf.engine().endScope();
}
```

## Future Research Directions

- **Memory-optimized Transfer Learning**: Implementing pre-trained model adaptation with limited browser memory
- **Comparative Optimization Visualization**: Side-by-side comparison of optimization algorithms (SGD, Adam, RMSProp)
- **Attention Mechanism Visualization**: Extension to transformer-based models with attention flow visualization

## Citation

If you use this visualization framework in your research, please cite:

```
@software{neural_network_viz,
  author = {Khan, Umer},
  title = {Neural Network Visualization: Interactive Decision Boundary Analysis},
  year = {2025},
  url = {https://github.com/umerkhan95/Neural-Network-visualisation}
}
```

## License

MIT License

## Plan for Increasing Repository Visibility

### Pain Points of Neural Network Visualizations

1. **Memory Management**: Handling memory efficiently is crucial, especially when using JavaScript-based tensor operations. The repository addresses this with multi-tier memory handling and deterministic tensor disposal to prevent memory leaks.
2. **Real-time Updates**: Ensuring that visualizations update in real-time during training can be challenging. The repository uses epoch-by-epoch visualization and custom callbacks to update the visualizations periodically.
3. **Compatibility with Different Datasets**: Supporting various datasets (e.g., XOR, circle classification, regression) requires flexible visualization components. The repository includes dataset-specific visualizations and parameter adjustments.
4. **Error Handling**: Managing errors and exceptions during visualization is important to maintain a smooth user experience. The repository implements fallback visualization methods and detailed error logging to handle issues gracefully.
5. **User Interface**: Providing an intuitive and responsive user interface for parameter manipulation and visualization rendering is essential. The repository uses React and Bootstrap to create a user-friendly interface with interactive controls and tooltips.

### Plan to Address Pain Points

1. **Improve Documentation**:
   - Add detailed explanations of the visualization components and their usage.
   - Include more examples and use cases to demonstrate the capabilities of the repository.
   - Provide step-by-step guides for setting up and using the repository.

2. **Add More Examples**:
   - Create additional examples showcasing different neural network architectures and datasets.
   - Include examples of common neural network tasks such as image classification, text classification, and reinforcement learning.

3. **Engage with the Community**:
   - Encourage contributions from the community by providing clear guidelines for contributing to the repository.
   - Actively participate in discussions and address issues raised by users.
   - Promote the repository on social media platforms and relevant forums to increase visibility.

4. **Optimize the Visualization Code**:
   - Continuously monitor and optimize the memory usage of the visualization components.
   - Implement advanced memory management techniques such as `tf.tidy`, explicit tensor disposal, scope-based memory management, and batch processing.
   - Ensure that the visualizations remain responsive and smooth, even when handling large datasets or complex models.

5. **Enhance the User Interface**:
   - Improve the visual appeal of the interface by using modern design principles and customizing the Bootstrap theme.
   - Ensure that the interface is fully responsive and works well on different screen sizes.
   - Add interactive tutorials or guided tours to help users understand how to use the application.
   - Provide real-time feedback during training, such as progress bars and visual cues.
   - Improve error handling by making error messages more user-friendly and informative.
   - Add a dark mode option to cater to users who prefer a darker interface.
   - Ensure that the interface is accessible to all users, including those with disabilities.
   - Implement a feedback mechanism to gather user input on the interface and identify areas for improvement.
