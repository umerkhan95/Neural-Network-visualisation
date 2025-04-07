import * as tf from '@tensorflow/tfjs';
import { NeuralNetworkParameters } from '../components/ParameterControls';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export interface TrainingCallback {
  onEpochEnd: (epoch: number, logs: { loss: number, accuracy: number }) => void;
  onTrainingComplete: () => void;
}

// Map our user-friendly activation function names to TensorFlow.js activation identifiers
const mapActivationFunction = (activation: string): ActivationIdentifier => {
  switch (activation) {
    case 'relu':
      return 'relu';
    case 'sigmoid':
      return 'sigmoid';
    case 'tanh':
      return 'tanh';
    default:
      return 'relu';
  }
};

export class NeuralNetworkModel {
  public model: tf.Sequential | null = null;
  private isTraining: boolean = false;
  private shouldStopTraining: boolean = false;
  
  constructor() {
    // Initialize empty model
    this.model = null;
  }

  public createModel(parameters: NeuralNetworkParameters): tf.Sequential {
    // Dispose previous model if exists to free memory
    if (this.model) {
      this.model.dispose();
    }

    // Create a sequential model
    const model = tf.sequential();
    
    // Add input layer and hidden layers
    for (let i = 0; i < parameters.layers; i++) {
      const neurons = parameters.neuronsPerLayer[i];
      const activation = mapActivationFunction(parameters.activationFunction);
      
      if (i === 0) {
        // Input layer
        model.add(tf.layers.dense({
          units: neurons,
          activation: activation,
          inputShape: [2], // Default to 2D input for XOR, circle, etc.
        }));
      } else {
        // Hidden layers
        model.add(tf.layers.dense({
          units: neurons,
          activation: activation,
        }));
      }
    }
    
    // Add output layer - default to binary classification
    model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
    }));
    
    // Compile the model
    model.compile({
      optimizer: tf.train.adam(parameters.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });
    
    this.model = model;
    return model;
  }

  public async trainOnXOR(parameters: NeuralNetworkParameters, callbacks: TrainingCallback): Promise<void> {
    if (!this.model) {
      this.createModel(parameters);
    }
    
    // XOR data - using tf.tidy for memory management
    const xorData = tf.tidy(() => {
      return {
        inputs: tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]),
        outputs: tf.tensor2d([[0], [1], [1], [0]]),
      };
    });
    
    this.isTraining = true;
    this.shouldStopTraining = false;
    
    try {
      // Train the model
      await this.model!.fit(xorData.inputs, xorData.outputs, {
        epochs: parameters.epochs,
        batchSize: parameters.batchSize,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            if (logs) {
              callbacks.onEpochEnd(epoch, {
                loss: logs.loss as number,
                accuracy: logs.acc as number,
              });
            }
            
            // Check if training should be stopped
            if (this.shouldStopTraining) {
              this.model!.stopTraining = true;
            }
            
            // Allow UI to update
            await tf.nextFrame();
          },
          onTrainEnd: () => {
            this.isTraining = false;
            callbacks.onTrainingComplete();
          },
        },
      });
    } finally {
      // Clean up tensors after training
      xorData.inputs.dispose();
      xorData.outputs.dispose();
    }
  }

  public async trainOnCircleData(parameters: NeuralNetworkParameters, callbacks: TrainingCallback): Promise<void> {
    if (!this.model) {
      this.createModel(parameters);
    }
    
    // Generate circle classification data
    const numSamples = 500;
    const data = this.generateCircleData(numSamples);
    
    this.isTraining = true;
    this.shouldStopTraining = false;
    
    try {
      // Train the model
      await this.model!.fit(data.inputs, data.outputs, {
        epochs: parameters.epochs,
        batchSize: parameters.batchSize,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            if (logs) {
              callbacks.onEpochEnd(epoch, {
                loss: logs.loss as number,
                accuracy: logs.acc as number,
              });
            }
            
            // Check if training should be stopped
            if (this.shouldStopTraining) {
              this.model!.stopTraining = true;
            }
            
            // Allow UI to update
            await tf.nextFrame();
          },
          onTrainEnd: () => {
            this.isTraining = false;
            callbacks.onTrainingComplete();
          },
        },
      });
    } finally {
      // Clean up tensors after training
      data.inputs.dispose();
      data.outputs.dispose();
    }
  }

  public async trainOnRegressionData(parameters: NeuralNetworkParameters, callbacks: TrainingCallback): Promise<void> {
    // Create a new model specifically for regression
    if (this.model) {
      this.model.dispose();
    }
    
    // Create a sequential model for regression
    const model = tf.sequential();
    
    // Add input layer and hidden layers
    for (let i = 0; i < parameters.layers; i++) {
      const neurons = parameters.neuronsPerLayer[i];
      const activation = mapActivationFunction(parameters.activationFunction);
      
      if (i === 0) {
        // Input layer
        model.add(tf.layers.dense({
          units: neurons,
          activation: activation,
          inputShape: [1], // 1D input for regression
        }));
      } else {
        // Hidden layers
        model.add(tf.layers.dense({
          units: neurons,
          activation: activation,
        }));
      }
    }
    
    // Add output layer for regression (no activation)
    model.add(tf.layers.dense({
      units: 1,
    }));
    
    // Compile the model for regression
    model.compile({
      optimizer: tf.train.adam(parameters.learningRate),
      loss: 'meanSquaredError',
      metrics: ['mse'],
    });
    
    this.model = model;
    
    // Generate regression data
    const numSamples = 200;
    const data = this.generateRegressionData(numSamples);
    
    this.isTraining = true;
    this.shouldStopTraining = false;
    
    try {
      // Train the model
      await this.model.fit(data.inputs, data.outputs, {
        epochs: parameters.epochs,
        batchSize: parameters.batchSize,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            if (logs) {
              // For regression, we use MSE instead of accuracy
              callbacks.onEpochEnd(epoch, {
                loss: logs.loss as number,
                accuracy: 1 - (logs.mse as number) / 10, // Normalize MSE to a 0-1 scale for visualization
              });
            }
            
            // Check if training should be stopped
            if (this.shouldStopTraining) {
              this.model!.stopTraining = true;
            }
            
            // Allow UI to update
            await tf.nextFrame();
          },
          onTrainEnd: () => {
            this.isTraining = false;
            callbacks.onTrainingComplete();
          },
        },
      });
    } finally {
      // Clean up tensors after training
      data.inputs.dispose();
      data.outputs.dispose();
    }
  }

  public stopTraining(): void {
    this.shouldStopTraining = true;
  }

  public isModelTraining(): boolean {
    return this.isTraining;
  }

  public predict(inputs: tf.Tensor): tf.Tensor {
    if (!this.model) {
      throw new Error('Model has not been created yet');
    }
    
    // Use tf.tidy to clean up intermediate tensors
    return tf.tidy(() => {
      return this.model!.predict(inputs) as tf.Tensor;
    });
  }

  // Add a cleanup method to dispose of the model
  public dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }

  private generateCircleData(numSamples: number): { inputs: tf.Tensor2D, outputs: tf.Tensor2D } {
    const points: number[][] = [];
    const labels: number[] = [];
    
    for (let i = 0; i < numSamples; i++) {
      // Generate random points in a square from -1 to 1
      const x = (Math.random() * 2) - 1;
      const y = (Math.random() * 2) - 1;
      
      // Calculate distance from origin
      const distanceFromOrigin = Math.sqrt(x * x + y * y);
      
      // Points inside the circle (radius 0.5) are class 1, outside are class 0
      const label = distanceFromOrigin < 0.5 ? 1 : 0;
      
      points.push([x, y]);
      labels.push(label);
    }
    
    // Use tf.tidy to automatically clean up tensors
    return tf.tidy(() => {
      return {
        inputs: tf.tensor2d(points),
        outputs: tf.tensor2d(labels, [numSamples, 1]),
      };
    });
  }

  private generateRegressionData(numSamples: number): { inputs: tf.Tensor2D, outputs: tf.Tensor2D } {
    const xs: number[] = [];
    const ys: number[] = [];
    
    for (let i = 0; i < numSamples; i++) {
      // Generate x values between -1 and 1
      const x = (Math.random() * 2) - 1;
      
      // Generate y = x^3 + noise
      const y = Math.pow(x, 3) + ((Math.random() * 0.2) - 0.1);
      
      xs.push(x);
      ys.push(y);
    }
    
    // Use tf.tidy to automatically clean up tensors
    return tf.tidy(() => {
      return {
        inputs: tf.tensor2d(xs, [numSamples, 1]),
        outputs: tf.tensor2d(ys, [numSamples, 1]),
      };
    });
  }
}

export default NeuralNetworkModel;
