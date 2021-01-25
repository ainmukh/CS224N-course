# Что это?
Выданное задание можно найти на странице курса: <a href='https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/assignments/a5_public.zip'>код</a> и <a href='https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/assignments/a5_updated.pdf'>условия</a>.

# Что сделала я?
Переписала код всех файлов. Некоторые функции самой модели, код которых мне не нравился, изменила полностью, остальные переписала с небольшими изменениями.
Функции и методы, идейно понятные, например <code>pad_sents_char</code> в файле <code>utils.py</code>, скопировала без изменений.
Скопированные участки кода в файлах обрамлены комментариями <code># копипаст</code> и <code># /копипаст</code>.

# Что представляет из себя модель?
Архитектура предложена в задании, менять ее не требовали.
И encoder и decoder – LSTM. Для слов, переводя которые модель выдает <code>\<unk></code> – CNN char-decoder.

# Что получилось?
<a href='https://github.com/ainmukh/CS224n-course/blob/main/a5/outputs/test_outputs.txt'>Результат</a> перевода <a href='https://github.com/ainmukh/CS224n-course/blob/main/a5/en_es_data/test.es'>текста</a> с испанского. 

<b>Corpus BLEU: 36.38971426678862</b>. По условию полный балл ставится при BLEU $$\geq{36}$$
