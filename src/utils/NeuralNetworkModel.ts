import * as tf from '@tensorflow/tfjs';
import { NeuralNetworkParameters } from '../components/ParameterControls';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export interface TrainingCallback {
  onEpochEnd: (epoch: number, logs: { loss: number, accuracy: number }) => void;
  onTrainingComplete?: () => void;
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
    
    // Define XOR problem
    const xorData = {
      inputs: tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]),
      outputs: tf.tensor2d([[0], [1], [1], [0]])
    };

    try {
      console.log('Starting XOR training with parameters:', parameters);
      
      // Create a custom loss history to track training progress
      const lossHistory: number[] = [];
      const accuracyHistory: number[] = [];
      
      // Process parameters
      const batchSize = 4;
      const epochs = parameters.epochs;
      const batchesPerRun = 5; // Process 5 epochs per fit call to avoid memory issues
      const totalRuns = Math.ceil(epochs / batchesPerRun);
      
      // Training loop - break up training into smaller chunks to avoid memory issues
      for (let run = 0; run < totalRuns; run++) {
        const currentEpoch = run * batchesPerRun;
        const remainingEpochs = Math.min(batchesPerRun, epochs - currentEpoch);
        
        if (remainingEpochs <= 0) break;
        
        // Use a separate scope for memory management
        tf.engine().startScope();
        
        // Custom callbacks for this batch of epochs
        const customCallbacks = {
          onEpochEnd: (epoch: number, logs: any) => {
            const actualEpoch = currentEpoch + epoch;
            if (callbacks?.onEpochEnd) {
              // Make sure we properly extract accuracy from TensorFlow.js logs
              // TF.js uses 'acc' instead of 'accuracy' in its logs
              const accuracy = typeof logs.acc !== 'undefined' ? logs.acc : 
                               typeof logs.accuracy !== 'undefined' ? logs.accuracy : 0;
              
              callbacks.onEpochEnd(actualEpoch, {
                loss: logs.loss || 0,
                accuracy: accuracy
              });
            }
            
            // Track loss and accuracy history
            lossHistory.push(logs.loss || 0);
            accuracyHistory.push(logs.acc || logs.accuracy || 0);
          }
        };
        
        // Train on a fresh copy of the data to avoid any tensor reuse issues
        const trainingInputs = xorData.inputs.clone();
        const trainingOutputs = xorData.outputs.clone();
        
        try {
          await this.model!.fit(trainingInputs, trainingOutputs, {
            batchSize,
            epochs: remainingEpochs,
            callbacks: customCallbacks
          });
        } catch (fitError) {
          console.error('Error during model.fit:', fitError);
          throw fitError;
        } finally {
          // Clean up tensors when we're done with them
          trainingInputs.dispose();
          trainingOutputs.dispose();
        }
        
        // End the scope to release any remaining tensors
        tf.engine().endScope();
        
        // Allow the event loop to process and clean up memory between batches
        await new Promise(resolve => setTimeout(resolve, 50));
        
        // Force a garbage collection
        tf.engine().startScope();
        tf.engine().endScope();
      }
      
      console.log('XOR training completed successfully');
      callbacks.onTrainingComplete?.();
    } catch (error) {
      console.error('Error during XOR training:', error);
      throw error;
    } finally {
      // Clean up tensors
      xorData.inputs.dispose();
      xorData.outputs.dispose();
      
      // Final cleanup
      tf.engine().startScope();
      tf.engine().endScope();
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
      console.log('Starting circle data training with parameters:', parameters);
      
      // Workaround for 'disposeNewTensors is not defined' issue
      // Instead of using fit directly with a large number of epochs,
      // we'll run fit multiple times with fewer epochs
      const batchSize = parameters.batchSize;
      const epochs = parameters.epochs;
      const batchesPerRun = 5; // Process 5 epochs per fit call to avoid memory issues
      const totalRuns = Math.ceil(epochs / batchesPerRun);
      
      for (let run = 0; run < totalRuns; run++) {
        const currentEpoch = run * batchesPerRun;
        const remainingEpochs = Math.min(batchesPerRun, epochs - currentEpoch);
        
        if (remainingEpochs <= 0) break;
        
        // Create a custom callback to handle epoch reporting
        const customCallbacks = {
          onEpochEnd: async (epoch: number, logs: any) => {
            const actualEpoch = currentEpoch + epoch;
            if (callbacks?.onEpochEnd) {
              // Make sure we properly extract accuracy from TensorFlow.js logs
              // TF.js uses 'acc' instead of 'accuracy' in its logs
              const accuracy = typeof logs.acc !== 'undefined' ? logs.acc : 
                               typeof logs.accuracy !== 'undefined' ? logs.accuracy : 0;
              
              callbacks.onEpochEnd(actualEpoch, {
                loss: logs.loss || 0,
                accuracy: accuracy
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
            callbacks.onTrainingComplete?.();
          },
        };
        
        await this.model!.fit(data.inputs, data.outputs, {
          batchSize,
          epochs: remainingEpochs,
          callbacks: customCallbacks
        });
        
        // Allow event loop to process and clean up memory
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      console.log('Circle data training completed successfully');
    } finally {
      // Clean up tensors after training
      data.inputs.dispose();
      data.outputs.dispose();
      
      // Force a garbage collection
      tf.tidy(() => {});
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
      console.log('Starting regression data training with parameters:', parameters);
      
      // Workaround for 'disposeNewTensors is not defined' issue
      // Instead of using fit directly with a large number of epochs,
      // we'll run fit multiple times with fewer epochs
      const batchSize = parameters.batchSize;
      const epochs = parameters.epochs;
      const batchesPerRun = 5; // Process 5 epochs per fit call to avoid memory issues
      const totalRuns = Math.ceil(epochs / batchesPerRun);
      
      for (let run = 0; run < totalRuns; run++) {
        const currentEpoch = run * batchesPerRun;
        const remainingEpochs = Math.min(batchesPerRun, epochs - currentEpoch);
        
        if (remainingEpochs <= 0) break;
        
        // Create a custom callback to handle epoch reporting
        const customCallbacks = {
          onEpochEnd: async (epoch: number, logs: any) => {
            const actualEpoch = currentEpoch + epoch;
            if (callbacks?.onEpochEnd) {
              // For regression, we use MSE instead of accuracy
              // Make sure we properly extract accuracy from TensorFlow.js logs
              // TF.js uses 'acc' instead of 'accuracy' in its logs
              const accuracy = typeof logs.acc !== 'undefined' ? logs.acc : 
                               typeof logs.accuracy !== 'undefined' ? logs.accuracy : 0;
              
              callbacks.onEpochEnd(actualEpoch, {
                loss: logs.loss || 0,
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
            callbacks.onTrainingComplete?.();
          },
        };
        
        await this.model!.fit(data.inputs, data.outputs, {
          batchSize,
          epochs: remainingEpochs,
          callbacks: customCallbacks
        });
        
        // Allow event loop to process and clean up memory
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      console.log('Regression data training completed successfully');
    } finally {
      // Clean up tensors after training
      data.inputs.dispose();
      data.outputs.dispose();
      
      // Force a garbage collection
      tf.tidy(() => {});
    }
  }

  public stopTraining(): void {
    this.shouldStopTraining = true;
  }

  public isModelTraining(): boolean {
    return this.isTraining;
  }

  public async predict(inputs: tf.Tensor): Promise<tf.Tensor> {
    if (!this.model) {
      throw new Error('Model not created yet');
    }

    try {
      // First approach: Try standard batch prediction with managed memory
      console.log('Attempting batch prediction...');
      
      // Ensure inputs are in the expected shape for the model
      const inputShape = this.model.inputs[0].shape;
      const batchSize = inputs.shape[0];
      
      console.log('Model expects input shape:', inputShape);
      console.log('Input tensor shape:', inputs.shape);
      
      // Check if shapes are compatible (ignoring batch dimension)
      if (inputShape[1] && inputs.shape[1] && inputShape[1] !== inputs.shape[1]) {
        throw new Error(`Input shape mismatch: model expects ${inputShape.slice(1)}, got ${inputs.shape.slice(1)}`);
      }
      
      // Direct prediction - we'll try this first as it's the simplest approach
      return this.model.predict(inputs) as tf.Tensor;
    } catch (error) {
      console.error('Batch prediction failed, trying simpler approach:', error);
      
      // Second approach: Skip reshape operations and use a more direct approach
      try {
        console.log('Trying simpler prediction approach...');
        
        // When model.predict fails, fall back to directly calling the output layer
        const lastLayer = this.model.layers[this.model.layers.length - 1];
        const inputLayer = this.model.layers[0];
        
        // Apply just input and output layers to avoid reshape errors
        const features = inputLayer.apply(inputs) as tf.Tensor;
        const prediction = lastLayer.apply(features) as tf.Tensor;
        
        // Clean up intermediate tensor
        features.dispose();
        
        return prediction;
      } catch (innerError) {
        console.error('All prediction methods failed:', innerError);
        
        // Return a dummy tensor of the right shape for visualization to continue
        // This tensor will contain zeros, which will display as a neutral color in visualization
        const outputShape = this.model.outputs[0].shape;
        const batchSize = inputs.shape[0];
        const outputSize = outputShape[1] || 1;
        
        // Create dummy output tensor
        return tf.zeros([batchSize, outputSize]);
      }
    }
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
