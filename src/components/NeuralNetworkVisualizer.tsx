import React, { useRef, useEffect } from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import * as tf from '@tensorflow/tfjs';

interface NeuralNetworkVisualizerProps {
  parameters?: {
    layers: number;
    neuronsPerLayer: number[];
  };
  weights?: tf.Tensor[];
}

const NeuralNetworkVisualizer: React.FC<NeuralNetworkVisualizerProps> = ({ 
  parameters = { layers: 2, neuronsPerLayer: [8, 8] },
  weights
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Draw the neural network architecture
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set canvas dimensions
    const width = canvas.width;
    const height = canvas.height;
    
    // Calculate the number of layers to display (input + hidden + output)
    const totalLayers = parameters.layers + 1; // +1 for output layer
    
    // Calculate spacing
    const layerSpacing = width / (totalLayers + 1);
    const maxNeurons = Math.max(...parameters.neuronsPerLayer, 1); // Output layer has 1 neuron for binary classification
    const neuronSpacing = Math.min(height / (maxNeurons + 1), 40);
    const neuronRadius = 10;
    
    // Draw each layer
    for (let layer = 0; layer < totalLayers; layer++) {
      // For the last layer (output), we use 1 neuron for binary classification
      const neuronsInLayer = layer < parameters.layers 
        ? parameters.neuronsPerLayer[layer] 
        : 1;
      
      const layerX = (layer + 1) * layerSpacing;
      const layerHeight = neuronsInLayer * neuronSpacing;
      const startY = (height - layerHeight) / 2 + neuronSpacing / 2;
      
      // Draw neurons in this layer
      for (let neuron = 0; neuron < neuronsInLayer; neuron++) {
        const neuronY = startY + neuron * neuronSpacing;
        
        // Draw connections to previous layer
        if (layer > 0) {
          const prevLayerX = layer * layerSpacing;
          const prevNeuronsInLayer = parameters.neuronsPerLayer[layer - 1];
          const prevLayerHeight = prevNeuronsInLayer * neuronSpacing;
          const prevStartY = (height - prevLayerHeight) / 2 + neuronSpacing / 2;
          
          // Draw connections to all neurons in previous layer
          for (let prevNeuron = 0; prevNeuron < prevNeuronsInLayer; prevNeuron++) {
            const prevNeuronY = prevStartY + prevNeuron * neuronSpacing;
            
            // Draw connection line
            ctx.beginPath();
            ctx.moveTo(prevLayerX, prevNeuronY);
            ctx.lineTo(layerX, neuronY);
            ctx.strokeStyle = 'rgba(150, 150, 150, 0.5)';
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
        
        // Draw neuron
        ctx.beginPath();
        ctx.arc(layerX, neuronY, neuronRadius, 0, Math.PI * 2);
        ctx.fillStyle = layer === totalLayers - 1 ? '#4CAF50' : '#2196F3';
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      
      // Add layer label
      ctx.fillStyle = '#000';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      let layerLabel = '';
      if (layer === 0) {
        layerLabel = 'Input Layer';
      } else if (layer === totalLayers - 1) {
        layerLabel = 'Output Layer';
      } else {
        layerLabel = `Hidden Layer ${layer}`;
      }
      ctx.fillText(layerLabel, layerX, height - 10);
    }
  }, [parameters]);
  
  return (
    <Card className="neural-network-visualizer mb-4">
      <Card.Header>Network Architecture</Card.Header>
      <Card.Body>
        <div className="network-canvas-container">
          <canvas 
            ref={canvasRef} 
            width={600} 
            height={300} 
            className="network-canvas"
          />
        </div>
        <div className="mt-3">
          <p className="text-center">
            <small>
              This visualization shows the structure of the neural network based on your parameters.
              <br />
              Blue circles represent neurons in input and hidden layers, green represents output neurons.
            </small>
          </p>
        </div>
      </Card.Body>
    </Card>
  );
};

export default NeuralNetworkVisualizer;
