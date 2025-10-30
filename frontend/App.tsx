
import React, { useState, useCallback } from 'react';
import { InterviewTopic, InterviewStatus } from './types';
import TopicSelection from './components/TopicSelection';
import InterviewSession from './components/InterviewSession';

const App: React.FC = () => {
  const [topic, setTopic] = useState<InterviewTopic | null>(null);
  const [status, setStatus] = useState<InterviewStatus>('not-started');

  const handleTopicSelect = useCallback((selectedTopic: InterviewTopic) => {
    setTopic(selectedTopic);
    setStatus('in-progress');
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4 font-sans">
      <div className="w-full max-w-2xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
            AI HR Interviewer
          </h1>
          <p className="text-gray-400 mt-2">
            Hone your skills with a technical interview.
          </p>
        </header>

        <main className="bg-gray-800 rounded-2xl shadow-2xl overflow-hidden">
          {status === 'not-started' && <TopicSelection onTopicSelect={handleTopicSelect} />}
          {status === 'in-progress' && topic && <InterviewSession topic={topic} />}
        </main>
        
        <footer className="text-center mt-8 text-gray-500 text-sm">
            <p>Ready to start your interview? Good luck!</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
