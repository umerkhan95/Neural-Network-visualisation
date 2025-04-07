import React from 'react';
import { Card, Form } from 'react-bootstrap';

interface DatasetSelectorProps {
  onDatasetChange: (dataset: string) => void;
  selectedDataset: string;
}

const DatasetSelector: React.FC<DatasetSelectorProps> = ({ 
  onDatasetChange,
  selectedDataset 
}) => {
  return (
    <Card className="dataset-selector mb-3">
      <Card.Header>Dataset Selection</Card.Header>
      <Card.Body>
        <Form.Group>
          <Form.Label>Select a dataset for training:</Form.Label>
          <Form.Control 
            as="select" 
            value={selectedDataset} 
            onChange={(e) => onDatasetChange(e.target.value)}
          >
            <option value="xor">XOR Problem</option>
            <option value="circle">Circle Classification</option>
            <option value="regression">Simple Regression</option>
            <option value="mnist">MNIST Digits (Subset)</option>
          </Form.Control>
        </Form.Group>
        
        <div className="dataset-description mt-3">
          {selectedDataset === 'xor' && (
            <div>
              <h5>XOR Problem</h5>
              <p>A classic binary classification problem that requires a neural network to learn the XOR (exclusive OR) function. This is a simple problem that demonstrates the need for hidden layers in neural networks.</p>
            </div>
          )}
          {selectedDataset === 'circle' && (
            <div>
              <h5>Circle Classification</h5>
              <p>A 2D classification problem where points inside a circle belong to one class and points outside belong to another. This demonstrates decision boundaries in neural networks.</p>
            </div>
          )}
          {selectedDataset === 'regression' && (
            <div>
              <h5>Simple Regression</h5>
              <p>A regression problem where the network learns to predict a continuous value based on input features. This demonstrates how neural networks can approximate functions.</p>
            </div>
          )}
          {selectedDataset === 'mnist' && (
            <div>
              <h5>MNIST Digits (Subset)</h5>
              <p>A simplified version of the MNIST handwritten digits dataset. This demonstrates how neural networks can recognize patterns in image data.</p>
            </div>
          )}
        </div>
      </Card.Body>
    </Card>
  );
};

export default DatasetSelector;
