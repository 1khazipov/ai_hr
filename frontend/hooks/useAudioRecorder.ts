import { useState, useRef, useCallback } from 'react';

interface AudioRecorderHook {
  isRecording: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<{ blob: Blob; mimeType: string }>;
}

export const useAudioRecorder = (): AudioRecorderHook => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async (): Promise<void> => {
    if (isRecording) {
      console.warn('Recording is already in progress.');
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Media Devices API not supported in this browser.');
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstart = () => {
        setIsRecording(true);
      };

      mediaRecorder.start();
    } catch (err) {
      console.error('Error starting recording:', err);
      throw new Error('Could not start recording. Please grant microphone permission.');
    }
  }, [isRecording]);

  const stopRecording = useCallback((): Promise<{ blob: Blob; mimeType: string }> => {
    return new Promise((resolve, reject) => {
      if (!mediaRecorderRef.current || !isRecording) {
        return reject('No active recording to stop.');
      }

      mediaRecorderRef.current.onstop = () => {
        const mimeType = mediaRecorderRef.current?.mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

        mediaRecorderRef.current?.stream.getTracks().forEach(track => track.stop());

        setIsRecording(false);
        resolve({ blob: audioBlob, mimeType });
      };

      mediaRecorderRef.current.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        setIsRecording(false);
        reject('An error occurred during recording.');
      };

      mediaRecorderRef.current.stop();
    });
  }, [isRecording]);

  return { isRecording, startRecording, stopRecording };
};
