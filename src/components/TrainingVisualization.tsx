import React from 'react';
import { Line } from 'react-chartjs-2';
import { Card } from 'react-bootstrap';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface TrainingVisualizationProps {
  lossHistory: number[];
  accuracyHistory: number[];
  currentEpoch: number;
}

const TrainingVisualization: React.FC<TrainingVisualizationProps> = ({ 
  lossHistory, 
  accuracyHistory,
  currentEpoch 
}) => {
  const epochs = Array.from({ length: lossHistory.length }, (_, i) => i + 1);

  const lossData = {
    labels: epochs,
    datasets: [
      {
        label: 'Loss',
        data: lossHistory,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.1,
      },
    ],
  };

  const accuracyData = {
    labels: epochs,
    datasets: [
      {
        label: 'Accuracy',
        data: accuracyHistory,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Training Progress',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Card className="training-visualization">
      <Card.Header>Training Progress (Epoch {currentEpoch})</Card.Header>
      <Card.Body>
        <div className="chart-container">
          <h5>Loss</h5>
          <Line data={lossData} options={options} />
        </div>
        <div className="chart-container mt-4">
          <h5>Accuracy</h5>
          <Line data={accuracyData} options={options} />
        </div>
      </Card.Body>
    </Card>
  );
};

export default TrainingVisualization;
