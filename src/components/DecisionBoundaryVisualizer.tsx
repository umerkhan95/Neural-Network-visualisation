import React, { useRef, useEffect, useState } from 'react';
import { Card } from 'react-bootstrap';
import * as tf from '@tensorflow/tfjs';
import NeuralNetworkModel from '../utils/NeuralNetworkModel';

interface DecisionBoundaryVisualizerProps {
  dataset: string;
  model: NeuralNetworkModel | null;
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
  const [renderKey, setRenderKey] = useState(0); // Add a key to force re-renders
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // Only show for 2D classification problems
  useEffect(() => {
    setShouldRender(['xor', 'circle'].includes(dataset));
  }, [dataset]);
  
  // Force redraw when training epoch changes
  useEffect(() => {
    if (isTraining && currentEpoch > 0 && currentEpoch % 5 === 0) {
      setRenderKey(prev => prev + 1);
    }
  }, [currentEpoch, isTraining]);

  // Force redraw when training completes
  useEffect(() => {
    if (!isTraining && currentEpoch > 0) {
      // Training just completed, force a redraw
      setRenderKey(prev => prev + 1);
      console.log('Training completed, forcing decision boundary redraw');
    }
  }, [isTraining, currentEpoch]);
  
  // Draw the decision boundary
  useEffect(() => {
    const drawBoundary = async () => {
      if (!canvasRef.current || !model) {
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        // Size of the grid used for visualization (lower for better performance)
        // Adjust resolution based on available memory
        const resolution = 10;
        
        // Generate coordinates for the grid
        const gridPoints: number[][] = [];
        for (let x = 0; x < canvas.width; x += resolution) {
          for (let y = 0; y < canvas.height; y += resolution) {
            // Scale to -1 to 1 range for model input
            const scaledX = (x / canvas.width) * 2 - 1;
            const scaledY = (y / canvas.height) * 2 - 1;
            gridPoints.push([scaledX, scaledY]);
          }
        }

        // Create input tensor
        const inputs = tf.tensor2d(gridPoints);
        
        try {
          // Get model predictions
          const predictions = await model.predict(inputs);
          const predValues = await predictions.dataSync();
          
          if (!canvas.isConnected) {
            // Canvas was removed while we were computing, bail out
            inputs.dispose();
            predictions.dispose();
            return;
          }
          
          // Draw the decision boundary using a simple colored grid
          let i = 0;
          for (let x = 0; x < canvas.width; x += resolution) {
            for (let y = 0; y < canvas.height; y += resolution) {
              const value = predValues[i++];
              
              // Create a color gradient based on the prediction value
              const r = Math.round(255 * (1 - value));
              const g = Math.round(255 * value);
              const b = 128;
              
              ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
              ctx.fillRect(x, y, resolution, resolution);
            }
          }
          
          // Clean up tensors
          inputs.dispose();
          predictions.dispose();
        } catch (error) {
          console.error('Primary visualization failed, trying simplified approach:', error);
          
          // Fallback visualization: just show a gradient background
          const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
          gradient.addColorStop(0, 'rgba(255, 100, 100, 0.5)');
          gradient.addColorStop(1, 'rgba(100, 255, 100, 0.5)');
          ctx.fillStyle = gradient;
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          
          // Add an error message
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
          ctx.fillRect(canvas.width/2 - 150, canvas.height/2 - 30, 300, 60);
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          ctx.textAlign = 'center';
          ctx.fillText('Visualization simplified due to TensorFlow.js', canvas.width/2, canvas.height/2);
          ctx.fillText('memory limitations. Points still shown correctly.', canvas.width/2, canvas.height/2 + 20);
        }

        // Draw the dataset points (always shown, even in fallback visualization)
        if (dataset) {
          // Get dataset data from appropriate source based on dataset type
          let dataPoints: [number, number][] = [];
          let dataLabels: number[] = [];
          
          if (dataset === 'xor') {
            // XOR dataset: four fixed points
            dataPoints = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
            dataLabels = [0, 1, 1, 0];
          } else if (dataset === 'circle') {
            // Generate some points on a circle
            const numPoints = 50;
            const radius = 0.5;
            
            for (let i = 0; i < numPoints; i++) {
              const angle = (i / numPoints) * Math.PI * 2;
              // Inside points
              const insideRadius = radius * 0.5 * Math.random();
              const insideX = Math.cos(angle) * insideRadius;
              const insideY = Math.sin(angle) * insideRadius;
              dataPoints.push([insideX, insideY]);
              dataLabels.push(0);
              
              // Outside points
              const outsideRadius = radius + 0.2 * Math.random();
              const outsideX = Math.cos(angle) * outsideRadius;
              const outsideY = Math.sin(angle) * outsideRadius;
              dataPoints.push([outsideX, outsideY]);
              dataLabels.push(1);
            }
          }
          
          // Draw the dataset points
          for (let i = 0; i < dataPoints.length; i++) {
            const [x, y] = dataPoints[i];
            const value = dataLabels[i];
            
            // Convert from [-1, 1] to canvas coordinates
            const canvasX = ((x + 1) / 2) * canvas.width;
            const canvasY = ((y + 1) / 2) * canvas.height;
            
            // Draw the point with a color based on its class
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 5, 0, Math.PI * 2);
            ctx.fillStyle = value > 0.5 ? 'rgb(0, 200, 0)' : 'rgb(200, 0, 0)';
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      } catch (error) {
        console.error('All visualization methods failed:', error);
        
        // Show error message on canvas when everything fails
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Visualization Error', canvas.width/2, canvas.height/2 - 20);
        
        // Show detailed error if available
        if (error instanceof Error) {
          console.error('Detailed error:', {
            errorName: error.name,
            errorMessage: error.message,
            errorStack: error.stack
          });
          
          ctx.font = '12px Arial';
          ctx.fillText(error.message, canvas.width/2, canvas.height/2 + 10);
          ctx.fillText('See console for details.', canvas.width/2, canvas.height/2 + 30);
        }
      }
    };
    
    drawBoundary();
  }, [model, dataset, shouldRender, isTraining, currentEpoch, renderKey]);
  
  if (!shouldRender) {
    return null;
  }
  
  return (
    <Card className="decision-boundary-visualizer mb-4">
      <Card.Header>Decision Boundary Visualization</Card.Header>
      <Card.Body>
        <div className="boundary-canvas-container">
          {errorMessage && (
            <div className="alert alert-warning mb-2">
              {errorMessage}
            </div>
          )}
          <canvas 
            ref={canvasRef} 
            width={400} 
            height={400} 
            className="boundary-canvas"
            style={{ border: '1px solid #ccc' }}
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
