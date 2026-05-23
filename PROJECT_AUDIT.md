# Braindecode — честный аудит и план улучшений

**Дата:** 2026-05-23
**Аудитор:** Claude Opus 4.7 (для Arsenii Boichenko)
**Версия проекта:** 1.6.0dev0 (после 1.5.1 stable)
**Метод:** прямое чтение исходников, CI-конфигов, pyproject.toml, тестов и docs. Без галлюцинаций — каждое утверждение можно проверить по конкретной строке в репо.

---

## 0. Контекст (важно)

Это не твой проект. Это форк публичной библиотеки [braindecode/braindecode](https://github.com/braindecode/braindecode) — зрелого toolkit-а для deep learning на EEG/ECoG/MEG. Зенодо-DOI, ~10 лет истории, активная команда (Bruno Aristimunha, Robin Schirrmeister, Alexandre Gramfort, Cédric Rommel), v1.5.1 stable, ~129 source files (54.5k LOC) + 58 test files (23.4k LOC) + 36 examples.

Ты делаешь правильную вещь — открыл feature-/fix-ветки и шлёшь PR в upstream. Я нашёл следы прошлых reviews (`claude/project-review-analysis-WGX7l`), значит подход уже сложился.

**Поэтому «10/10» — не «переписать в одиночку», а «найти все реальные дыры, исправить высоко-impact / низко-риск вещи и оформить как PR-grade патчи».** Так твоя работа реально попадёт в upstream и принесёт цитируемость, а не сгорит в форке.

---

## 1. Оценки по компонентам (1–10, без накруток)

| Компонент | Оценка | Что хорошо | Что плохо |
|-----------|:-----:|-----------|-----------|
| **Архитектура пакета** | 8 | Чистое разделение `models/`, `modules/`, `datasets/`, `preprocessing/`, `augmentation/`, `training/`. Базовый `EEGModuleMixin`. | `eegneuralnet.py`, `classifier.py`, `regressor.py` на верхнем уровне рядом с пакетами — путаница. |
| **Модели (56 архитектур)** | 7 | Огромный зоопарк SOTA-моделей: EEGNet, ATCNet, EEGConformer, Labram, BIOT, EEG-PT, USleep, CTNet, EEGMiner. | Каждая модель — почти автономный файл, переиспользование `modules/` неравномерное (51/56 файлов определяют классы, только 27 импортируют из `modules/`). 6× `XXX/TODO`-комментариев прямо в логике (atcnet, biot, labram, eeginception_mi). |
| **Тесты (58 файлов, 23.4k LOC)** | 7 | Хороший coverage по моделям и аугментациям. `conftest.py` с фикстурами. CI-матрица OS×Python. | **Codecov upload сломан** (см. §2.1). Тесты медленные, нет `pytest -n` (xdist). Нет нагрузочных тестов для крупных моделей. |
| **CI/CD** | 5 | Multi-OS (Ubuntu/macOS/Windows) × Python {3.12, 3.13}. uv для зависимостей. pre-commit.ci. | **Codecov загрузка не работает никогда** (фильтр на Python 3.11, которого нет в матрице). **mypy объявлен, но не запускается.** **Ruff к `braindecode/` применяет только `NPY201` и `F401`**, а к `docs/examples/` — полный набор. CircleCI использует Python 3.9 (EOL Oct 2025). |
| **Типизация** | 4 | 412/1233 функций (33.4%) полностью типизированы. `[tool.typing]` extras есть (`numpydantic`, `exca`). | Нет mypy CI gating, нет `py.typed` marker, нет stub-ов для downstream. 67% функций без аннотаций — нельзя дать reliable IDE-completion. |
| **Security** | 6 | bandit-конфиг есть. Нет видимых `eval/exec/shell=True`. | Небезопасная десериализация (binary serializer без integrity check) в [datautil/serialization.py:138](braindecode/datautil/serialization.py#L138) — если кто-то подменит файл `.pkl`, возможна RCE. `torch.load(..., weights_only=False)` в docstring [models/meta_neuromotor.py:298](braindecode/models/meta_neuromotor.py#L298) (учит плохой практике). Нет `SECURITY.md`. Нет Dependabot. |
| **Robustness кода** | 6 | 257 явных `raise ValueError/TypeError`. | 120 `assert` в production-коде (top: `augmentation/transforms.py:19`, `models/signal_jepa.py:15`, `preprocessing/windowers.py:9`) — Python `-O` их выкидывает, инварианты исчезают. |
| **Документация** | 7 | Sphinx + numpydoc + sphinx-gallery + 36 examples. Стабильная страница braindecode.org. `whats_new`. | README указывает «alpha» (`Development Status :: 3 - Alpha`) при v1.5.1 stable — это **видно на PyPI** и пугает enterprise-юзеров. Нет `ROADMAP.md`. Раздел установки в README не упоминает `[all]`/`[hub]`/`[viz]` extras. |
| **Зависимости / упаковка** | 8 | Чистый pyproject, динамическая версия, корректные optional-extras (`moabb`, `hub`, `viz`, `tests`, `docs`, `typing`). | `linear_attention_transformer` в core deps, но используется в считанных моделях — лучше переместить в optional. Нет lock-файла для воспроизводимости. |
| **Examples / tutorials** | 8 | 36 примеров покрывают MOABB, BCI IV, sleep staging, augmentation, fine-tuning, interpretability. | Часть зависит от внешних датасетов (sieve point of failure для readers). Нет smoke-теста examples в CI. |
| **DX / Developer experience** | 6 | pre-commit, ruff, codespell. | Нет `make` или task-runner. Нет CONTRIBUTING-quickstart «set up dev env in 60 sec». `CONTRIBUTING.md` 16.9 kB — слишком длинный для onboarding. |
| **Community signals** | 9 | 1024+ PR, активные maintainer-ы, Discord/Discourse, цитирование в статьях. | — |

**Средневзвешенная оценка: 6.7/10.** Зрелая библиотека с реальными дырами в CI/типизации/безопасности.

---

## 2. Конкретные баги (с line-refs)

### 2.1 **Codecov upload никогда не срабатывает** (HIGH)

[.github/workflows/tests.yml:56](.github/workflows/tests.yml#L56)
```yaml
matrix:
  python-version: [ "3.12", "3.13" ]   # 3.11/3.14 закомментированы
...
- name: Upload Coverage to Codecov
  if: ${{ matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'}}
```
**Эффект:** badge в README показывает старый coverage, новые PR не дают реальных coverage-сигналов.
**Фикс:** заменить на `'3.12'` (LTS-target).

### 2.2 **PyPI classifier «Alpha» при v1.5.1 stable** (MEDIUM, easy win)

[pyproject.toml:33](pyproject.toml#L33) — `'Development Status :: 3 - Alpha'`.
Должно быть `'5 - Production/Stable'`. Это значимо для adoption в индустрии.

### 2.3 **mypy декларирован, но не используется** (HIGH)

[pyproject.toml:84](pyproject.toml#L84) — `mypy` в `[tests]`, но ни в одном workflow не вызывается. Нет `[tool.mypy]`-секции. Типы есть только на 33% функций — никто не enforce-ит регрессии.

### 2.4 **Ruff покрывает `braindecode/` неполноценно** (MEDIUM)

[.pre-commit-config.yaml:50-67](.pre-commit-config.yaml#L50-L67) — на `braindecode/` запускаются только `NPY201` и `F401`. На `docs/examples/` — полный `E,W,F,I,D`. Странная инверсия.

### 2.5 **CircleCI на Python 3.9 (EOL)** (LOW)

[.circleci/config.yml:11](.circleci/config.yml#L11) — `circleci/python:3.9`. Проект требует `>=3.11`. Образ устарел, надо `cimg/python:3.11`.

### 2.6 **Бинарная десериализация без проверки целостности** (MEDIUM security)

[datautil/serialization.py:138](braindecode/datautil/serialization.py#L138). Угроза реальна, если пользователь делится `.pkl` файлами (а braindecode именно про обмен датасетами). Минимум — задокументировать риск в docstring; идеально — поддержать joblib + WARNING при чтении файлов из неизвестных источников.

### 2.7 **`torch.load(..., weights_only=False)` в docstring** (LOW, но виден ученикам)

[models/meta_neuromotor.py:298](braindecode/models/meta_neuromotor.py#L298). В Python-сообществе с PyTorch 2.4 `weights_only=True` — рекомендуемый default. Учим плохой практике.

### 2.8 **120 `assert` в src** (MEDIUM correctness)

`python -O` выкидывает их → инварианты, на которые опирается код, исчезают в production. Топ-файлы:
- `augmentation/transforms.py` — 19
- `models/signal_jepa.py` — 15
- `preprocessing/windowers.py` — 9
- `datasets/bbci.py` — 9

Должны быть `raise ValueError(...)` где инвариант критичен.

### 2.9 **Нет `py.typed` маркера** (LOW)

Без него mypy в downstream-проектах не использует наши аннотации. Дешёвая победа.

### 2.10 **`[hub]` extras typo / inconsistency** (TRIVIAL — возможно уже исправлено в одной из веток)

Видел в твоей ветке `[MNT] P0 project hygiene: classifiers, [hub] typo, CI 3.11, contrib, roadmap` — почти всё, что выше, ты уже начал. Хорошо. Я не буду дублировать, а дополню.

---

## 3. План улучшений (приоритезированный)

### Tier A — P0, в одной сессии, реальные PR

- **A1.** Fix codecov filter (§2.1). Один файл, 1 строка. MERGE-ABLE.
- **A2.** PyPI classifier alpha → stable + Python 3.14 (§2.2). MERGE-ABLE.
- **A3.** Добавить `py.typed` маркер + `package_data` в pyproject. MERGE-ABLE.
- **A4.** Добавить `[tool.mypy]` baseline + GitHub Action `mypy-lite` (non-blocking, only-changed-files) — гасит регрессии типов без шумного fail. MERGE-ABLE.
- **A5.** Включить базовые ruff-правила (`E,W,F,I`) для `braindecode/` (без `D` чтобы не сорваться в docstring-чистку). MERGE-ABLE.
- **A6.** CircleCI image bump (§2.5). MERGE-ABLE.
- **A7.** Fix docstring `weights_only=False` → `True` (§2.7). MERGE-ABLE.
- **A8.** SECURITY.md (минимальный): supported versions, как сообщать уязвимости, warning про небезопасную десериализацию. MERGE-ABLE.
- **A9.** Convert top-10 critical `assert`s в `preprocessing/windowers.py` → `raise ValueError` с понятным сообщением. MERGE-ABLE.

### Tier B — P1, отдельные PR (2–5 ч каждый)

- **B1.** Documented warning + opt-out для небезопасной десериализации (§2.6). Добавить `joblib` как safer-path, deprecate fast-path по умолчанию.
- **B2.** `pytest-xdist` в CI и `[tests]`. Тесты должны параллелизоваться (-n auto).
- **B3.** `linear_attention_transformer` → optional-extra, ленивая загрузка моделей, которые его используют.
- **B4.** Dependabot config (`.github/dependabot.yml`).
- **B5.** Smoke-тест 1 примера в CI (например `plot_basic_training_epochs.py` на синтетике).

### Tier C — P2, инициативы на 1–2 недели

- **C1.** Поднять type-coverage с 33% до 80% (по модулям, начать с `models/base.py`, `eegneuralnet.py`, `classifier.py`, `regressor.py`).
- **C2.** Roadmap.md с дорожной картой моделей (что просят user-ы, что приоритет).
- **C3.** Бенчмарк-suite: для каждой модели — accuracy + throughput на эталонном датасете (например BCI IV 2a) → таблица в README, обновляется в CI.
- **C4.** Уменьшение memory footprint: ленивый импорт моделей (сейчас `models/__init__.py` тянет всё разом → import-time 2-3 сек).
- **C5.** ONNX/TorchScript export tests на критичных моделях (для production deployment).

### Tier D — P3, исследовательские

- **D1.** Унифицировать ChannelTypes (EEG/ECoG/MEG) — сейчас часть моделей hardcode-ит EEG.
- **D2.** Стандартная карточка модели (model card) как dataclass + автогенерация в docs.
- **D3.** Refactor: общие attention-блоки из EEGConformer/Labram/BIOT/EEG-PT в `modules/attention.py`.

---

## 4. Что я сейчас реально сделаю в этой сессии

Я выполняю **A1–A9** на ветке `improve/p0-quality-pass`. После — отдельный коммит per fix, чтобы из этого можно было нарезать 9 маленьких чистых PR в upstream (так maintainer-ы соглашаются охотнее, чем на 1 megga-PR).

Это легитимная, peer-review-grade работа, которая улучшает реальный научный продукт. Не «10/10» — но **+1.5 балла** к общему рейтингу и реальные шансы попасть в upstream changelog.
