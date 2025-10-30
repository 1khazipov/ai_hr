import { Message } from '../types';

const CHAT_API_URL = 'http://10.10.32.12:8080/chat';
const STT_API_URL = 'http://10.10.32.12:8081/transcribe';
const TTS_API_URL = 'http://10.10.32.12:8082/synthesize';


export async function getChatResponse(messages: Message[]): Promise<string> {
  try {
    const response = await fetch(CHAT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to get chat response' }));
      throw new Error(`Chat API error: ${response.status} ${response.statusText} - ${errorData.detail}`);
    }
    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('Network or CORS error in getChatResponse:', error);
    throw new Error('Failed to connect to the chat service. Is it running on port 8000 with CORS configured?');
  }
}

export async function transcribeAudio(audioBlob: Blob, mimeType: string): Promise<string> {
  const formData = new FormData();
  const extension = mimeType.split('/')[1].split(';')[0];
  const filename = `interview_response.${extension}`;
  
  formData.append('file', audioBlob, filename);

  try {
    const response = await fetch(STT_API_URL, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
       const errorData = await response.json().catch(() => ({ detail: 'Failed to transcribe audio' }));
      throw new Error(`STT API error: ${response.status} ${response.statusText} - ${errorData.detail}`);
    }
    const data = await response.json();
    return data.transcription;
  } catch (error) {
    console.error('Network or CORS error in transcribeAudio:', error);
    throw new Error('Failed to connect to the transcription service. Is it running on port 8081 with CORS configured?');
  }
}

export async function synthesizeSpeech(text: string): Promise<Blob> {
  try {
    const response = await fetch(TTS_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, speaker: 'baya' }),
    });
     if (!response.ok) {
      const errorText = await response.text();
      console.error('TTS API Error Response Text:', errorText);
      throw new Error(`TTS API error: ${response.status} ${response.statusText} - ${errorText}`);
    }
    return await response.blob();
  } catch (error) {
    console.error('Network or CORS error in synthesizeSpeech:', error);
    throw new Error('Failed to connect to the speech synthesis service. Is it running on port 8082 with CORS configured?');
  }
}
