@echo off

REM Creating directories that needed by container
mkdir ".\ai_model\cahya_bert-base-indonesian-522M"

REM Start Docker Compose
docker-compose -f compose.yml up -d --build