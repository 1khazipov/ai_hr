
import React from 'react';
import { InterviewTopic } from '../types';

interface TopicSelectionProps {
  onTopicSelect: (topic: InterviewTopic) => void;
}

const topics: { id: InterviewTopic; name: string; description: string }[] = [
  { id: 'frontend', name: 'Frontend', description: 'React, TypeScript, CSS, and web performance.' },
  { id: 'backend', name: 'Backend', description: 'APIs, databases, system design, and scalability.' },
  { id: 'ml', name: 'Machine Learning', description: 'Algorithms, data structures, and model deployment.' },
];

const TopicCard: React.FC<{ topic: typeof topics[0], onSelect: () => void }> = ({ topic, onSelect }) => (
    <button
        onClick={onSelect}
        className="w-full text-left p-6 bg-gray-700/50 rounded-lg border border-gray-600 hover:bg-gray-700 hover:border-blue-400 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
    >
        <h3 className="text-xl font-semibold text-white">{topic.name}</h3>
        <p className="text-gray-400 mt-1">{topic.description}</p>
    </button>
);


const TopicSelection: React.FC<TopicSelectionProps> = ({ onTopicSelect }) => {
  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold text-center mb-6">Choose your interview topic</h2>
      <div className="space-y-4">
        {topics.map((topic) => (
          <TopicCard key={topic.id} topic={topic} onSelect={() => onTopicSelect(topic.id)} />
        ))}
      </div>
    </div>
  );
};

export default TopicSelection;
