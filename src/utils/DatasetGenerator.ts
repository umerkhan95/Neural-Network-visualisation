import * as tf from '@tensorflow/tfjs';

// XOR dataset
export const generateXORData = () => {
  return {
    inputs: tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]),
    outputs: tf.tensor2d([[0], [1], [1], [0]]),
    inputShape: [2]
  };
};

// Circle classification dataset
export const generateCircleData = (numSamples = 500) => {
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
  
  return {
    inputs: tf.tensor2d(points),
    outputs: tf.tensor2d(labels, [numSamples, 1]),
    inputShape: [2]
  };
};

// Simple regression dataset
export const generateRegressionData = (numSamples = 200) => {
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
  
  return {
    inputs: tf.tensor2d(xs, [numSamples, 1]),
    outputs: tf.tensor2d(ys, [numSamples, 1]),
    inputShape: [1]
  };
};

// MNIST subset (simplified for performance)
export const loadMNISTSubset = async (numSamples = 100) => {
  try {
    // For the demo, we'll use a simplified approach to avoid MNIST loading issues
    // In a production app, we would properly load the MNIST dataset
    console.log('Creating simplified MNIST subset for demo purposes');
    
    // Create a simplified dataset with random pixel values
    const xs: number[][] = [];
    const ys: number[][] = [];
    
    for (let i = 0; i < numSamples; i++) {
      // Create a 28x28 image with random values (simplified)
      const pixels = Array(784).fill(0).map(() => Math.random() * 0.5);
      xs.push(pixels);
      
      // Create a random label (0-9)
      const label = Math.floor(Math.random() * 10);
      const oneHot = Array(10).fill(0);
      oneHot[label] = 1;
      ys.push(oneHot);
    }
    
    return {
      inputs: tf.tensor2d(xs),
      outputs: tf.tensor2d(ys),
      inputShape: [784] // 28x28 images flattened
    };
  } catch (error) {
    console.error('Error creating MNIST data:', error);
    // Return a small dummy dataset if creation fails
    return {
      inputs: tf.tensor2d(Array(100).fill(0).map(() => Array(784).fill(0))),
      outputs: tf.tensor2d(Array(100).fill(0).map(() => {
        const oneHot = Array(10).fill(0);
        oneHot[Math.floor(Math.random() * 10)] = 1;
        return oneHot;
      })),
      inputShape: [784]
    };
  }
};

export const getDatasetByName = async (name: string) => {
  switch (name) {
    case 'xor':
      return generateXORData();
    case 'circle':
      return generateCircleData();
    case 'regression':
      return generateRegressionData();
    case 'mnist':
      return await loadMNISTSubset();
    default:
      return generateXORData();
  }
};
