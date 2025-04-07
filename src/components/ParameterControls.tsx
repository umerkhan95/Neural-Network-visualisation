import React, { useState } from 'react';
import { Form, Row, Col, Card } from 'react-bootstrap';
import InfoTooltip, { educationalContent } from './EducationalTooltips';

interface ParameterControlsProps {
  onParametersChange: (parameters: NeuralNetworkParameters) => void;
}

export interface NeuralNetworkParameters {
  layers: number;
  neuronsPerLayer: number[];
  learningRate: number;
  epochs: number;
  batchSize: number;
  activationFunction: string;
}

const ParameterControls: React.FC<ParameterControlsProps> = ({ onParametersChange }) => {
  const [parameters, setParameters] = useState<NeuralNetworkParameters>({
    layers: 2,
    neuronsPerLayer: [8, 8],
    learningRate: 0.01,
    epochs: 50,
    batchSize: 32,
    activationFunction: 'relu'
  });

  const handleLayersChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newLayers = parseInt(event.target.value);
    const newNeuronsPerLayer = [...parameters.neuronsPerLayer];
    
    // Adjust neurons per layer array based on new layer count
    if (newLayers > parameters.layers) {
      // Add new layers with default neuron count
      for (let i = parameters.layers; i < newLayers; i++) {
        newNeuronsPerLayer.push(8);
      }
    } else if (newLayers < parameters.layers) {
      // Remove excess layers
      newNeuronsPerLayer.splice(newLayers);
    }

    const newParameters = {
      ...parameters,
      layers: newLayers,
      neuronsPerLayer: newNeuronsPerLayer
    };
    
    setParameters(newParameters);
    onParametersChange(newParameters);
  };

  const handleNeuronsChange = (layerIndex: number, event: React.ChangeEvent<HTMLInputElement>) => {
    const neurons = parseInt(event.target.value);
    const newNeuronsPerLayer = [...parameters.neuronsPerLayer];
    newNeuronsPerLayer[layerIndex] = neurons;
    
    const newParameters = {
      ...parameters,
      neuronsPerLayer: newNeuronsPerLayer
    };
    
    setParameters(newParameters);
    onParametersChange(newParameters);
  };

  const handleParameterChange = (param: keyof NeuralNetworkParameters, value: any) => {
    const newParameters = {
      ...parameters,
      [param]: value
    };
    
    setParameters(newParameters);
    onParametersChange(newParameters);
  };

  return (
    <Card className="parameter-controls">
      <Card.Header>Network Parameters</Card.Header>
      <Card.Body>
        <Form>
          <Form.Group as={Row} className="mb-3">
            <Form.Label column sm={4} className="parameter-label">
              Number of Layers
              <InfoTooltip id="layers" content={educationalContent.layers} />
            </Form.Label>
            <Col sm={8}>
              <Form.Control 
                type="range" 
                min="1" 
                max="5" 
                value={parameters.layers} 
                onChange={handleLayersChange}
              />
              <Form.Text>{parameters.layers}</Form.Text>
            </Col>
          </Form.Group>

          {parameters.neuronsPerLayer.map((neurons, index) => (
            <Form.Group as={Row} className="mb-3" key={`layer-${index}`}>
              <Form.Label column sm={4} className="parameter-label">
                Neurons in Layer {index + 1}
                <InfoTooltip id={`neurons-${index}`} content={educationalContent.neurons} />
              </Form.Label>
              <Col sm={8}>
                <Form.Control 
                  type="range" 
                  min="1" 
                  max="32" 
                  value={neurons} 
                  onChange={(e) => handleNeuronsChange(index, e as React.ChangeEvent<HTMLInputElement>)}
                />
                <Form.Text>{neurons}</Form.Text>
              </Col>
            </Form.Group>
          ))}

          <Form.Group as={Row} className="mb-3">
            <Form.Label column sm={4} className="parameter-label">
              Learning Rate
              <InfoTooltip id="learning-rate" content={educationalContent.learningRate} />
            </Form.Label>
            <Col sm={8}>
              <Form.Control 
                type="range" 
                min="0.001" 
                max="0.1" 
                step="0.001" 
                value={parameters.learningRate} 
                onChange={(e) => handleParameterChange('learningRate', parseFloat(e.target.value))}
              />
              <Form.Text>{parameters.learningRate}</Form.Text>
            </Col>
          </Form.Group>

          <Form.Group as={Row} className="mb-3">
            <Form.Label column sm={4} className="parameter-label">
              Epochs
              <InfoTooltip id="epochs" content={educationalContent.epochs} />
            </Form.Label>
            <Col sm={8}>
              <Form.Control 
                type="number" 
                min="1" 
                max="200" 
                value={parameters.epochs} 
                onChange={(e) => handleParameterChange('epochs', parseInt(e.target.value))}
              />
            </Col>
          </Form.Group>

          <Form.Group as={Row} className="mb-3">
            <Form.Label column sm={4} className="parameter-label">
              Batch Size
              <InfoTooltip id="batch-size" content={educationalContent.batchSize} />
            </Form.Label>
            <Col sm={8}>
              <Form.Control 
                as="select" 
                value={parameters.batchSize} 
                onChange={(e) => handleParameterChange('batchSize', parseInt(e.target.value))}
              >
                <option value="1">1</option>
                <option value="8">8</option>
                <option value="16">16</option>
                <option value="32">32</option>
                <option value="64">64</option>
              </Form.Control>
            </Col>
          </Form.Group>

          <Form.Group as={Row} className="mb-3">
            <Form.Label column sm={4} className="parameter-label">
              Activation Function
              <InfoTooltip id="activation" content={educationalContent.activation} />
            </Form.Label>
            <Col sm={8}>
              <Form.Control 
                as="select" 
                value={parameters.activationFunction} 
                onChange={(e) => handleParameterChange('activationFunction', e.target.value)}
              >
                <option value="relu">ReLU</option>
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
              </Form.Control>
            </Col>
          </Form.Group>
        </Form>
      </Card.Body>
    </Card>
  );
};

export default ParameterControls;
