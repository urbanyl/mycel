FROM python:3.12-slim

LABEL org.opencontainers.image.title="mycel" \
      org.opencontainers.image.description="MYCEL reference front end: lexer, ledger parser, SI unit and axis algebra, static checker, CLI." \
      org.opencontainers.image.source="https://github.com/urbanyl/mycel" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/mycel
COPY pyproject.toml README.md LICENSE ./
COPY mycel ./mycel
RUN pip install --no-compile .

WORKDIR /work
ENTRYPOINT ["mycel"]
CMD ["--help"]
