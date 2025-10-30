export type InterviewTopic = 'frontend' | 'backend' | 'ml';
export type InterviewStatus = 'not-started' | 'in-progress';

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}
