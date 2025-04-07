import React, { useState } from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';
import { InfoCircle } from 'react-bootstrap-icons';

interface TooltipProps {
  content: string;
  id: string;
}

export const InfoTooltip: React.FC<TooltipProps> = ({ content, id }) => {
  return (
    <OverlayTrigger
      placement="right"
      overlay={
        <Tooltip id={`tooltip-${id}`}>
          {content}
        </Tooltip>
      }
    >
      <span className="tooltip-wrapper">
        <InfoCircle className="info-icon" size={16} />
      </span>
    </OverlayTrigger>
  );
};

export const educationalContent = {
  layers: "Layers are the building blocks of neural networks. The input layer receives data, hidden layers process it, and the output layer produces predictions. More layers can learn more complex patterns but may lead to overfitting.",
  
  neurons: "Neurons are the basic computational units in neural networks. Each neuron receives inputs, applies weights and an activation function, and outputs a value. More neurons can capture more complex patterns.",
  
  learningRate: "Learning rate controls how much the model's weights are adjusted during training. Higher values can lead to faster learning but may cause instability. Lower values provide more stable learning but may take longer to converge.",
  
  epochs: "An epoch is one complete pass through the entire training dataset. More epochs allow the model to learn more from the data, but too many can lead to overfitting.",
  
  batchSize: "Batch size determines how many samples are processed before the model's weights are updated. Larger batches provide more stable gradients but require more memory. Smaller batches introduce more noise but can help escape local minima.",
  
  activation: "Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. ReLU is fast and works well for many problems. Sigmoid outputs values between 0-1, useful for binary classification. Tanh outputs values between -1 and 1.",
  
  xorProblem: "The XOR (exclusive OR) problem is a classic example that demonstrates why neural networks need hidden layers. A single-layer network cannot solve XOR, but adding a hidden layer makes it possible.",
  
  circleClassification: "Circle classification demonstrates how neural networks can learn non-linear decision boundaries. Points inside a circle belong to one class, while points outside belong to another.",
  
  regression: "Regression problems involve predicting continuous values rather than discrete classes. The neural network learns to approximate a function that maps inputs to continuous outputs.",
  
  mnist: "MNIST is a dataset of handwritten digits used to train image classification models. It's a standard benchmark in machine learning and demonstrates how neural networks can recognize patterns in image data.",
  
  loss: "Loss measures how far the model's predictions are from the actual values. During training, the model tries to minimize this value. A decreasing loss curve indicates the model is learning.",
  
  accuracy: "Accuracy measures the percentage of correct predictions. For classification problems, it shows how often the model predicts the right class. For regression, it's derived from how close predictions are to actual values.",
  
  decisionBoundary: "The decision boundary is the line (or curve in 2D, surface in 3D) that separates different classes in the input space. Neural networks learn these boundaries during training."
};

export default InfoTooltip;
