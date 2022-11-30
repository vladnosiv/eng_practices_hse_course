# eng_practices_hse_course

## Install Dev

```bash
poetry install
```

## Install Prod

```bash
poetry install --without dev
```

## Config for PyPI

```bash
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi <YOUR-
```

## Build and publish

```bash
poetry publish --build --repository test-pypi
```

## Package link
```
https://test.pypi.org/project/vnd-eng-practices-course/
```

## Install with pip
```bash
pip install -i https://test.pypi.org/simple/ vnd-eng-practices-course
```

## Install with poetry
```bash
poetry add --source test-pypi vnd-eng-practices-course
```

## Run tests
```bash
python3 src/sanity_test.py
```