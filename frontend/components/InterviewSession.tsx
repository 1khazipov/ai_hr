import React, { useState, useEffect, useRef, useCallback } from 'react';
import { InterviewTopic, Message } from '../types';
import { getChatResponse, transcribeAudio, synthesizeSpeech } from '../services/api';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import MessageBubble from './MessageBubble';
import Spinner from './Spinner';
import { MicrophoneIcon } from './icons/MicrophoneIcon';
import { StopIcon } from './icons/StopIcon';
import { SendIcon } from './icons/SendIcon';
import { CogIcon } from './icons/CogIcon';
import { DownloadIcon } from './icons/DownloadIcon';

interface InterviewSessionProps {
  topic: InterviewTopic;
}

const InterviewSession: React.FC<InterviewSessionProps> = ({ topic }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [textInput, setTextInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastRecording, setLastRecording] = useState<{ blob: Blob; mimeType: string } | null>(null);

  const { isRecording, startRecording, stopRecording } = useAudioRecorder();
  const audioPlayerRef = useRef<HTMLAudioElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const topicMap: Record<InterviewTopic, string> = {
    frontend: 'Frontend-разработка',
    backend: 'Backend-разработка',
    ml: 'Машинное обучение',
  };

  useEffect(() => {
    const topicRussian = topicMap[topic];
    const prompt = `Ты — опытный HR-интервьюер, проводящий техническое собеседование на позицию по ${topicRussian}. Начни собеседование с фразы: "Здравствуйте! Я рад провести с вами собеседование на позицию по ${topicRussian}. Давайте начнем с основного вопроса: расскажите о своем опыте работы." и дождись ответа пользователя. Задавай по одному вопросу за раз. Твои вопросы должны быть краткими и по теме. Предлагай технические задачи и вопросы. Основывай свои следующие вопросы на предыдущих ответах пользователя. Отвечай только на русском языке от лица девушки.`;
    setSystemPrompt(prompt);
  }, [topic]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  useEffect(() => {
    const player = audioPlayerRef.current;
    if (player) {
      const onPlay = () => setIsSpeaking(true);
      const onEnded = () => setIsSpeaking(false);
      player.addEventListener('play', onPlay);
      player.addEventListener('ended', onEnded);
      return () => {
        player.removeEventListener('play', onPlay);
        player.removeEventListener('ended', onEnded);
      };
    }
  }, []);

  const speak = useCallback(async (text: string) => {
    if (!text || !text.trim()) {
      console.warn("Skipping speech synthesis for empty text.");
      setIsProcessing(false);
      return;
    }
    
    setIsProcessing(true);
    try {
      setError(null);
      const audioBlob = await synthesizeSpeech(text);
      const audioUrl = URL.createObjectURL(audioBlob);
      if (audioPlayerRef.current) {
        audioPlayerRef.current.src = audioUrl;
        await audioPlayerRef.current.play();
      }
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred during speech synthesis.';
      console.error("Speaking error:", errorMessage);
      setError(`Failed to synthesize speech. ${errorMessage}`);
      setIsProcessing(false);
    }
  }, []);
  
  const startInterview = useCallback(async () => {
    if (!systemPrompt) return;
    
    setInterviewStarted(true);
    setIsProcessing(true);
    setError(null);
    
    const initialMessages: Message[] = [{ role: 'system', content: systemPrompt }];
    
    try {
      const aiResponse = await getChatResponse(initialMessages);
      setMessages([{ role: 'assistant', content: aiResponse }]);
      await speak(aiResponse);
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
      console.error("Failed to start interview:", errorMessage);
      setError(`Could not start the interview. ${errorMessage}`);
      setInterviewStarted(false);
      setIsProcessing(false);
    }
  }, [systemPrompt, speak]);

  const processUserResponse = useCallback(async (userText: string) => {
    if (!userText.trim()) {
      setError("Cannot send an empty message.");
      return;
    }

    setIsProcessing(true);
    setError(null);

    const newMessages: Message[] = [...messages, { role: 'user', content: userText }];
    setMessages(newMessages);

    const historyForApi: Message[] = [{ role: 'system', content: systemPrompt }, ...newMessages];

    try {
      const aiResponse = await getChatResponse(historyForApi);
      setMessages(prev => [...prev, { role: 'assistant', content: aiResponse }]);
      await speak(aiResponse);
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
      console.error("Processing error:", errorMessage);
      setError(`There was an issue getting a response. ${errorMessage}`);
      setIsProcessing(false);
    }
  }, [messages, systemPrompt, speak]);

  const handleStopRecording = async () => {
    try {
      setError(null);
      setIsProcessing(true);
      const recordingResult = await stopRecording();
      setLastRecording(recordingResult); 
      
      const transcribedText = await transcribeAudio(recordingResult.blob, recordingResult.mimeType);
      
      if (transcribedText.trim()) {
        await processUserResponse(transcribedText);
      } else {
        setError("Простите, я ничего не расслышал. Пожалуйста, попробуйте еще раз.");
        setIsProcessing(false);
      }
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred during recording.';
      console.error("Recording/transcription error:", errorMessage);
      setError(`Ошибка записи: ${errorMessage}`);
      setIsProcessing(false);
    }
  };
  
  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    processUserResponse(textInput);
    setTextInput('');
  };

  const handleStartRecording = async () => {
      try {
          setError(null);
          await startRecording();
      } catch (e) {
          const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
          setError(errorMessage);
      }
  };

  const handleDownloadRecording = () => {
    if (!lastRecording) return;
    const { blob, mimeType } = lastRecording;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const extension = mimeType.split('/')[1].split(';')[0];
    a.download = `debug_recording.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-[80vh]">
      <audio ref={audioPlayerRef} onEnded={() => setIsProcessing(false)} />
      <div className="p-4 border-b border-gray-700 flex justify-between items-center gap-4">
        <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold">Тема: {topicMap[topic]}</h2>
            {lastRecording && (
                <button 
                    onClick={handleDownloadRecording}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-600 hover:bg-gray-500 rounded-md"
                    title="Скачать последнюю запись для отладки"
                >
                    <DownloadIcon className="h-4 w-4" />
                    Скачать запись
                </button>
            )}
        </div>
        <button onClick={() => setShowSettings(!showSettings)} className="p-2 rounded-full hover:bg-gray-700 flex-shrink-0">
          <CogIcon className="h-6 w-6" />
        </button>
      </div>
      
      {showSettings && (
        <div className="p-4 bg-gray-900/50">
          <label htmlFor="system-prompt" className="block text-sm font-medium text-gray-400 mb-2">Системный промпт:</label>
          <textarea
            id="system-prompt"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white text-sm"
            rows={6}
            disabled={isProcessing || messages.length > 0}
          />
        </div>
      )}

      <div className="flex-1 p-6 space-y-6 overflow-y-auto">
        {messages.map((msg, index) => (
           <MessageBubble 
              key={index} 
              message={msg} 
              isSpeaking={isSpeaking && msg.role === 'assistant' && index === messages.length - 1}
            />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {!interviewStarted && (
        <div className="p-4 flex flex-col items-center justify-center">
            <button
                onClick={startInterview}
                disabled={isProcessing}
                className="px-6 py-3 bg-blue-600 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors"
            >
                {isProcessing ? <Spinner /> : 'Начать собеседование'}
            </button>
        </div>
      )}

      {interviewStarted && (
        <>
        {error && <div className="p-4 text-center text-red-400 bg-red-900/50">{error}</div>}
        <div className="p-4 border-t border-gray-700">
            <form className="flex items-center space-x-4" onSubmit={handleSendText}>
              <input 
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Или введите ваш ответ..."
                disabled={isProcessing || isRecording}
              />
              <button type="submit" className="p-3 bg-blue-600 rounded-full hover:bg-blue-700 disabled:bg-gray-500" disabled={isProcessing || isRecording}>
                <SendIcon className="h-6 w-6 text-white"/>
              </button>
              <button 
                type="button" 
                onClick={isRecording ? handleStopRecording : handleStartRecording} 
                className={`p-4 rounded-full transition-colors ${isRecording ? 'bg-red-600 animate-pulse' : 'bg-blue-600 hover:bg-blue-700'} disabled:bg-gray-500`}
                disabled={isProcessing}
              >
                {isRecording ? <StopIcon className="h-6 w-6 text-white" /> : <MicrophoneIcon className="h-6 w-6 text-white" />}
              </button>
            </form>
        </div>
        </>
      )}
    </div>
  );
};

export default InterviewSession;