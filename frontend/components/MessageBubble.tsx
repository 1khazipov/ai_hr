
import React from 'react';
import { Message } from '../types';

interface MessageBubbleProps {
  message: Message;
  isSpeaking?: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isSpeaking = false }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-md lg:max-w-lg px-5 py-3 rounded-2xl shadow ${
          isUser
            ? 'bg-blue-600 rounded-br-lg'
            : 'bg-gray-700 rounded-bl-lg'
        }`}
      >
        <p className="text-white whitespace-pre-wrap">{message.content}</p>
        {isSpeaking && (
            <div className="flex items-center space-x-1 mt-2">
                <span className="text-xs text-gray-400">Speaking</span>
                <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                <div className="w-1 h-1 bg-gray-400 rounded-full animate-pulse"></div>
            </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
