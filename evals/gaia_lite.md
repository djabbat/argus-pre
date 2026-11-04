# ARGUS-OS1 — GAIA-Lite eval (V1.4)

Лёгкий, воспроизводимый eval с реальными инструментами (как GAIA L1, но меньше).
Формат: `## <название>` + строки `task:` / `answer:` / `type:`.
`type`: contains (answer — подстрока финального ответа) | file (<path>) | file_contains (<path>,<text>) | memory (<fact_key>).

## T1 web-title
task: Открой https://example.com и назови заголовок страницы
answer: Example Domain
type: contains

## T2 create-file
task: Создай файл eval_result.txt в рабочей директории со словом HELLO
answer: eval_result.txt,HELLO
type: file_contains

## T3 memory-city
task: Запомни: мой город Тбилиси. Потом скажи, какой у меня город
answer: Тбилиси
type: contains

## T4 shell-echo
task: Выполни команду echo HELLO_ARGUS и скажи её вывод
answer: HELLO_ARGUS
type: contains

## T5 web-search
task: Найди в интернете, что означает аббревиатура LLM, и дай краткий ответ
answer: LLM
type: contains
