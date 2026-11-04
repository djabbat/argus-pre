# ARGUS-OS1 — GAIA-L1 subset eval (V1.6)

Задачи уровня GAIA L1 (многошаговые, с реальными инструментами: web/file/shell/memory).
Формат: `## <название>` + `task:` / `answer:` / `type:`.
type: contains | file | file_contains | memory

## T1 un-countries
task: Сколько государств входит в ООН? Найди в интернете и назови число
answer: 193
type: contains

## T2 uname-arch
task: Выполни команду uname -m и скажи архитектуру процессора
answer: x86_64
type: contains

## T3 create-report
task: Создай файл report.txt с текстом GAIA L1 SUBSET в рабочей директории
answer: report.txt,GAIA L1 SUBSET
type: file_contains

## T4 remember-date
task: Запомни: дата эксперимента ARGUS 2026-10-02. Потом назови её
answer: 2026-10-02
type: contains

## T5 shell-multiply
task: Сколько будет 15 умножить на 13? Посчитай через shell (python3 или expr) и назови число
answer: 195
type: contains
