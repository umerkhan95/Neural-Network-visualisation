import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Button, Card, Alert } from 'react-bootstrap';
import NeuralNetworkVisualizer from './components/NeuralNetworkVisualizer';
import ParameterControls, { NeuralNetworkParameters } from './components/ParameterControls';
import TrainingVisualization from './components/TrainingVisualization';
import DatasetSelector from './components/DatasetSelector';
import DecisionBoundaryVisualizer from './components/DecisionBoundaryVisualizer';
import NeuralNetworkModel from './utils/NeuralNetworkModel';
import { getDatasetByName } from './utils/DatasetGenerator';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

const App: React.FC = () => {
  const [parameters, setParameters] = useState<NeuralNetworkParameters>({
    layers: 2,
    neuronsPerLayer: [8, 8],
    learningRate: 0.01,
    epochs: 50,
    batchSize: 32,
    activationFunction: 'relu'
  });
  
  const [selectedDataset, setSelectedDataset] = useState('xor');
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [showAlert, setShowAlert] = useState(false);

  // Use refs to maintain instance between renders
  const modelRef = useRef<NeuralNetworkModel | null>(null);

  // Initialize model on component mount
  useEffect(() => {
    modelRef.current = new NeuralNetworkModel();
    return () => {
      // Clean up TensorFlow memory when component unmounts
      if (modelRef.current) {
        modelRef.current.dispose();
      }
    };
  }, []);

  const handleParametersChange = (newParameters: NeuralNetworkParameters) => {
    setParameters(newParameters);
    // We don't recreate the model immediately, only when training starts
  };

  const handleDatasetChange = (dataset: string) => {
    setSelectedDataset(dataset);
    // Reset training metrics when dataset changes
    setLossHistory([]);
    setAccuracyHistory([]);
    setCurrentEpoch(0);
    setStatusMessage(`Selected dataset: ${dataset}`);
    setShowAlert(true);
    
    // Hide alert after 3 seconds
    setTimeout(() => {
      setShowAlert(false);
    }, 3000);
  };

  const handleStartTraining = async () => {
    if (!modelRef.current) return;
    
    try {
      setIsTraining(true);
      setStatusMessage('Preparing dataset...');
      setShowAlert(true);
      
      // Reset training metrics
      setLossHistory([]);
      setAccuracyHistory([]);
      setCurrentEpoch(0);
      
      // Get dataset
      const dataset = await getDatasetByName(selectedDataset);
      
      // Create model based on current parameters
      modelRef.current.createModel(parameters);
      
      setStatusMessage('Training started...');
      
      // Set up callbacks for visualization updates
      const callbacks = {
        onEpochEnd: (epoch: number, logs: { loss: number, accuracy: number }) => {
          setCurrentEpoch(epoch + 1);
          setLossHistory(prev => [...prev, logs.loss]);
          setAccuracyHistory(prev => [...prev, logs.accuracy]);
          setStatusMessage(`Training: Epoch ${epoch + 1}/${parameters.epochs} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${logs.accuracy.toFixed(4)}`);
        },
        onTrainingComplete: () => {
          setIsTraining(false);
          setStatusMessage('Training completed!');
          // Keep alert visible for training completion
        }
      };
      
      // Start training based on selected dataset
      switch (selectedDataset) {
        case 'xor':
          await modelRef.current.trainOnXOR(parameters, callbacks);
          break;
        case 'circle':
          await modelRef.current.trainOnCircleData(parameters, callbacks);
          break;
        case 'regression':
          await modelRef.current.trainOnRegressionData(parameters, callbacks);
          break;
        case 'mnist':
          // MNIST training would be implemented here
          setStatusMessage('MNIST training not fully implemented in this demo');
          setIsTraining(false);
          break;
        default:
          await modelRef.current.trainOnXOR(parameters, callbacks);
      }
    } catch (error) {
      console.error('Training error:', error);
      setStatusMessage(`Error during training: ${error}`);
      setIsTraining(false);
    }
  };

  const handleStopTraining = () => {
    if (modelRef.current) {
      modelRef.current.stopTraining();
      setStatusMessage('Training stopped by user');
    }
  };

  return (
    <Container fluid className="app-container py-4">
      <Row className="mb-4">
        <Col>
          <h1 className="text-center">Neural Network Visualization</h1>
          <p className="text-center lead">
            An interactive tool to understand how neural networks learn
          </p>
          {showAlert && (
            <Alert 
              variant="info" 
              onClose={() => setShowAlert(false)} 
              dismissible
            >
              {statusMessage}
            </Alert>
          )}
        </Col>
      </Row>

      <Row>
        <Col md={4}>
          <DatasetSelector 
            onDatasetChange={handleDatasetChange}
            selectedDataset={selectedDataset}
          />
          
          <ParameterControls onParametersChange={handleParametersChange} />
          
          <Card className="mt-3">
            <Card.Body>
              <Button 
                variant={isTraining ? "danger" : "primary"} 
                size="lg" 
                onClick={isTraining ? handleStopTraining : handleStartTraining}
                className="w-100"
                disabled={isTraining && currentEpoch >= parameters.epochs}
              >
                {isTraining ? "Stop Training" : "Start Training"}
              </Button>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={8}>
          <NeuralNetworkVisualizer 
            parameters={{
              layers: parameters.layers,
              neuronsPerLayer: parameters.neuronsPerLayer
            }}
          />
          
          {/* Decision boundary visualization for classification problems */}
          <DecisionBoundaryVisualizer 
            dataset={selectedDataset}
            model={modelRef.current?.model}
            isTraining={isTraining}
            currentEpoch={currentEpoch}
          />
          
          <TrainingVisualization 
            lossHistory={lossHistory}
            accuracyHistory={accuracyHistory}
            currentEpoch={currentEpoch}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default App;
