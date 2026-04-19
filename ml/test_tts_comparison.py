"""
Скрипт для тестирования старой и новой реализации TTS на больших текстах с английскими словами.

Старая модель: одна русская Silero TTS, без специальной обработки английских слов.
Новая модель: одна и та же русская Silero TTS, но текст проходит через normalize_text
из tts.py (транслитерация английских слов + проговаривание чисел).
"""
import time
from pathlib import Path

import soundfile as sf
import torch
from num2words import num2words
import re

from tts import normalize_text as normalize_text_new

# Тестовые тексты с английскими словами
TEST_TEXTS = [
    {
        "name": "Технический текст с терминологией",
        "text": """
        Современные технологии искусственного интеллекта и machine learning 
        открывают новые возможности для автоматизации бизнес-процессов. 
        Компании используют deep learning модели для анализа больших данных, 
        что позволяет улучшить customer experience и повысить эффективность работы. 
        Важным аспектом является интеграция API различных сервисов и использование 
        cloud computing решений для масштабирования инфраструктуры. 
        Современные разработчики работают с frameworks такими как TensorFlow, 
        PyTorch и scikit-learn для создания интеллектуальных систем.
        """
    },
    {
        "name": "Бизнес-текст с англицизмами",
        "text": """
        Наша компания специализируется на предоставлении IT-решений для крупного бизнеса. 
        Мы предлагаем комплексный подход к digital transformation, включающий консалтинг, 
        разработку custom software и внедрение enterprise систем. Наш team состоит из 
        опытных специалистов в области data science, cybersecurity и cloud infrastructure. 
        Мы используем agile методологии и best practices индустрии для обеспечения 
        высокого качества deliverables. Клиенты получают full support на всех этапах 
        проекта, от initial planning до post-deployment maintenance.
        """
    },
    {
        "name": "Научный текст с международными терминами",
        "text": """
        В современной науке особое внимание уделяется interdisciplinary исследованиям, 
        которые объединяют знания из различных областей. Quantum computing и 
        artificial intelligence представляют собой cutting-edge технологии, 
        способные революционизировать многие сферы человеческой деятельности. 
        Исследователи работают над созданием robust algorithms, которые могут 
        обрабатывать complex datasets и находить hidden patterns в данных. 
        Важную роль играет open source движение, которое способствует 
        collaboration между учеными по всему миру и ускоряет scientific progress.
        """
    }
]


def normalize_text_old(text: str) -> str:
    """Нормализация для старой версии (только числа)"""
    return re.sub(r'\d+', lambda m: num2words(int(m.group(0)), lang='ru'), text)


def normalize_ru_chunk(text: str) -> str:
    """
    Подготовка RU-куска для Silero RU TTS:
    - схлопываем пробелы/переводы строк
    - конвертируем числа
    - добавляем завершающую пунктуацию (иначе RU модель иногда падает)
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""
    # Если в куске нет ни одной буквы/цифры, не отправляем его в RU модель
    # (иначе модель может падать на чистой пунктуации типа ',.')
    if not re.search(r"[А-Яа-яЁё0-9]", text):
        return ""
    text = normalize_text_old(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def synthesize_with_old_model(text: str, speaker: str = 'baya', sample_rate: int = 48000):
    """
    Синтез речи используя старую модель (только русская, без английской поддержки).
    Имитирует работу старой версии tts.py
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем только русскую модель
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v4_ru'
    )
    model.to(device)
    
    # Нормализуем текст (только числа)
    normalized_text = normalize_text_old(text)
    
    # Синтезируем (английские слова будут проигнорированы или произнесены неправильно)
    audio_tensor = model.apply_tts(
        text=normalized_text,
        speaker=speaker,
        sample_rate=sample_rate
    )
    
    return audio_tensor.cpu().numpy(), sample_rate


def synthesize_with_new_model(text: str, speaker: str = 'baya', sample_rate: int = 48000):
    """
    Синтез речи используя НОВУЮ реализацию:
    - используется та же русская Silero TTS
    - но текст предварительно нормализуется через normalize_text из tts.py
      (английские слова транслитерируются, числа проговариваются по-русски).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v4_ru',
    )
    model.to(device)

    normalized_text = normalize_text_new(text)

    audio_tensor = model.apply_tts(
        text=normalized_text,
        speaker=speaker,
        sample_rate=sample_rate,
    )

    return audio_tensor.cpu().numpy(), sample_rate


def split_ru_en_segments(text: str):
    """Теперь не используется в новой модели, оставлено для совместимости."""
    return [('ru', text)]


def analyze_text(text: str):
    """Анализ текста: подсчет английских слов и их позиций"""
    en_words = re.findall(r'[A-Za-z]+', text)
    return {
        "total_chars": len(text),
        "en_words_count": len(en_words),
        "en_words": en_words,
        "has_english": len(en_words) > 0
    }


def main():
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ СТАРОЙ И НОВОЙ РЕАЛИЗАЦИИ TTS")
    print("=" * 80)
    print()
    
    # Создаем директорию для результатов
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    results_summary = []
    
    for idx, test_case in enumerate(TEST_TEXTS, 1):
        print(f"\n{'='*80}")
        print(f"ТЕСТ {idx}: {test_case['name']}")
        print(f"{'='*80}")
        
        text = test_case['text'].strip()
        analysis = analyze_text(text)
        
        print(f"\nАнализ текста:")
        print(f"  - Всего символов: {analysis['total_chars']}")
        print(f"  - Английских слов: {analysis['en_words_count']}")
        print(f"  - Примеры английских слов: {', '.join(analysis['en_words'][:10])}")
        print()
        
        # Тест старой модели
        print("🔴 Тестирование СТАРОЙ модели (только русская)...")
        start_time = time.time()
        try:
            audio_old, sr_old = synthesize_with_old_model(text, speaker='baya')
            old_time = time.time() - start_time
            old_duration = len(audio_old) / sr_old
            
            # Сохраняем аудио
            old_file = output_dir / f"test_{idx}_old_model.wav"
            sf.write(str(old_file), audio_old, sr_old)
            
            print(f"  ✓ Успешно синтезировано")
            print(f"  - Время синтеза: {old_time:.2f} сек")
            print(f"  - Длительность аудио: {old_duration:.2f} сек")
            print(f"  - Файл сохранен: {old_file}")
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            old_time = None
            old_duration = None
            old_file = None
        
        print()
        
        # Тест новой модели
        print("🟢 Тестирование НОВОЙ модели (русская + английская)...")
        start_time = time.time()
        try:
            audio_new, sr_new = synthesize_with_new_model(text, speaker='baya')
            new_time = time.time() - start_time
            new_duration = len(audio_new) / sr_new
            
            # Сохраняем аудио
            new_file = output_dir / f"test_{idx}_new_model.wav"
            sf.write(str(new_file), audio_new, sr_new)
            
            print(f"  ✓ Успешно синтезировано")
            print(f"  - Время синтеза: {new_time:.2f} сек")
            print(f"  - Длительность аудио: {new_duration:.2f} сек")
            print(f"  - Файл сохранен: {new_file}")
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            new_time = None
            new_duration = None
            new_file = None
        
        # Сохраняем результаты
        results_summary.append({
            "test_name": test_case['name'],
            "en_words_count": analysis['en_words_count'],
            "old_model": {
                "file": str(old_file) if old_file else None,
                "duration": old_duration,
                "time": old_time
            },
            "new_model": {
                "file": str(new_file) if new_file else None,
                "duration": new_duration,
                "time": new_time
            }
        })
        
        print()
    
    # Итоговая сводка
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 80)
    print()
    
    for result in results_summary:
        print(f"📝 {result['test_name']}")
        print(f"   Английских слов: {result['en_words_count']}")
        if result['old_model']['file']:
            print(f"   🔴 Старая модель: {result['old_model']['file']} "
                  f"({result['old_model']['duration']:.2f} сек, "
                  f"синтез: {result['old_model']['time']:.2f} сек)")
        if result['new_model']['file']:
            print(f"   🟢 Новая модель: {result['new_model']['file']} "
                  f"({result['new_model']['duration']:.2f} сек, "
                  f"синтез: {result['new_model']['time']:.2f} сек)")
        print()
    
    print(f"\nВсе результаты сохранены в директории: {output_dir.absolute()}")
    print("\nРекомендация: Прослушайте оба варианта для каждого теста и сравните")
    print("качество произношения английских слов.")


if __name__ == "__main__":
    main()

