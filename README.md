# Beeline Campaign Copilot

Последняя собранная версия демонстрационного сайта: рекомендации кампаний,
аудитории, каналы, расходы и результаты пилотов. В комплекте находятся
HTML-интерфейс, Python-сервер, агент и необходимые CSV.

Этот архив содержит прежний агент, с которым сайт был собран и запущен.
Новый агент из «the best one 11.zip» ещё не подключён к интерфейсу.
Данные синтетические; сервер использует упрощённую демонстрационную среду.

## Публикация через GitHub и Render

1. Распакуйте архив. Создайте репозиторий на https://github.com/new.
2. Через Add file → Upload files загрузите содержимое распакованной папки,
   включая папку backend вместе с её содержимым. Сохраните изменения через
   Commit changes. В корне репозитория должны лежать index.html и requirements.txt.
   Загружать сам ZIP вместо файлов не нужно.
3. На https://dashboard.render.com выберите New → Web Service.
4. Подключите GitHub и выберите созданный репозиторий.
5. Укажите Language: Python 3. Поле Root Directory оставьте пустым.
6. Build Command: pip install -r requirements.txt
7. Start Command: python backend/server.py
8. Выберите подходящий тариф хостинга и нажмите Deploy/Create Web Service.
9. После успешного запуска Render выдаст адрес https://…onrender.com.
   Откройте его: сайт и API будут работать на одном адресе.

Сервер в этом архиве слушает 0.0.0.0 и использует переменную PORT,
которую задаёт хостинг. GitHub Pages не выполняет серверный Python-код,
поэтому для этого приложения требуется Web Service.

Документация: https://render.com/docs/web-services

## Локальный запуск

Требуется Python 3.10 или новее. Из папки с requirements.txt выполните:

```powershell
python -m pip install -r requirements.txt
$env:HOST = "127.0.0.1"
python backend/server.py
```

Откройте http://127.0.0.1:8001/. Остановка: Ctrl+C.

API: GET /api/recommendations. Каждый запрос запускает демо-расчёт агента.
Сайт не отправляет сообщения абонентам и не создаёт официальный submission.csv.
