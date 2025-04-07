import React, { useRef, useEffect, useState } from 'react';
import { Card } from 'react-bootstrap';
import * as tf from '@tensorflow/tfjs';

interface DecisionBoundaryVisualizerProps {
  dataset: string;
  model: any;
  isTraining: boolean;
  currentEpoch: number;
}

const DecisionBoundaryVisualizer: React.FC<DecisionBoundaryVisualizerProps> = ({
  dataset,
  model,
  isTraining,
  currentEpoch
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [shouldRender, setShouldRender] = useState(false);
  
  // Only show for 2D classification problems
  useEffect(() => {
    setShouldRender(['xor', 'circle'].includes(dataset));
  }, [dataset]);
  
  // Draw the decision boundary
  useEffect(() => {
    if (!canvasRef.current || !model || !shouldRender) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Draw grid of predictions
    const resolution = 50; // Number of points in each dimension
    const gridSize = width / resolution;
    
    // Create a grid of points
    const xs = tf.tidy(() => {
      const points = [];
      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          // Convert from pixel space to problem space (-1 to 1)
          const x = (i / resolution) * 2 - 1;
          const y = (j / resolution) * 2 - 1;
          points.push([x, y]);
        }
      }
      return tf.tensor2d(points);
    });
    
    // Get predictions for all points
    tf.tidy(() => {
      try {
        const preds = model.predict(xs);
        const values = preds.dataSync();
        
        // Draw each point with color based on prediction
        for (let i = 0; i < resolution; i++) {
          for (let j = 0; j < resolution; j++) {
            const index = i * resolution + j;
            const value = values[index]; // Prediction value (0 to 1)
            
            // Map value to color (blue for 0, red for 1)
            const r = Math.floor(value * 255);
            const b = Math.floor((1 - value) * 255);
            const g = 100; // Some green to make it not too dark
            
            // Draw rectangle for this point
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.5)`;
            ctx.fillRect(i * gridSize, j * gridSize, gridSize, gridSize);
          }
        }
        
        // Draw dataset-specific elements
        if (dataset === 'xor') {
          // Draw XOR points
          const points = [[0, 0], [0, 1], [1, 0], [1, 1]];
          const labels = [0, 1, 1, 0];
          
          points.forEach((point, index) => {
            // Convert from problem space to pixel space
            const x = ((point[0] + 1) / 2) * width;
            const y = ((point[1] + 1) / 2) * height;
            
            // Draw point
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = labels[index] === 1 ? 'red' : 'blue';
            ctx.fill();
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.stroke();
          });
        } else if (dataset === 'circle') {
          // Draw circle boundary
          ctx.beginPath();
          const centerX = width / 2;
          const centerY = height / 2;
          const radius = width * 0.25; // Circle with radius 0.5 in problem space
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      } catch (error) {
        console.error('Error predicting decision boundary:', error);
      }
    });
    
    // Clean up tensors
    xs.dispose();
    
  }, [model, dataset, shouldRender, isTraining, currentEpoch]);
  
  if (!shouldRender) {
    return null;
  }
  
  return (
    <Card className="decision-boundary-visualizer mb-4">
      <Card.Header>Decision Boundary Visualization</Card.Header>
      <Card.Body>
        <div className="boundary-canvas-container">
          <canvas 
            ref={canvasRef} 
            width={400} 
            height={400} 
            className="boundary-canvas"
          />
        </div>
        <div className="mt-3">
          <p className="text-center">
            <small>
              This visualization shows how the neural network classifies different regions of the input space.
              <br />
              Blue regions are classified as 0, red regions as 1. The visualization updates as training progresses.
            </small>
          </p>
        </div>
      </Card.Body>
    </Card>
  );
};

export default DecisionBoundaryVisualizer;
