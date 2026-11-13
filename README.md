# Запуск лабы (Python 3, Windows / macOS / Linux)

ВАЖНО: запускать нужно на ВАШЕМ ноутбуке — хэши и замеры привязаны к вашему железу (mp.cpu_count()).
Закройте браузер/мессенджеры, подключите питание, не трогайте ноутбук во время замеров.

```
python task0_cal.py --id=ВАШ_ID
python task1_falsesharing.py
python task2_scaling.py
python task3_roofline_gemm.py
python task4_scheduling.py
python make_report.py --id=ВАШ_ID
```

Получится `lab_report_ВАШ_ID.md` — его и сдавайте (таблицы, 5 строк VERIFY_HASH, ответы, код).

Если скрипт не определил кэш/ядра автоматически, задайте вручную (до запуска task0), например:
- Linux/macOS: `L1D_KB=48 L2_KB=1280 L3_KB=12288 LINE=64 PPHYS=6 python task0_cal.py --id=...`
- Windows PowerShell: `$env:PPHYS=6; $env:L1D_KB=48; python task0_cal.py --id=...`

Флаг `--quick` — только для проверки, что всё запускается (хэши от него недействительны).
Phase 3/4 используют PPHYS (физические ядра). Весь прогон занимает порядка 10–30+ минут.
