# Urban ML Platform

Machine learning platform for urban intelligence.

## Running with Docker

Build the Docker image:

```bash
docker build -t urban-ml-platform .
```

Run the container:

```bash
docker run --rm -p 8000:8000 urban-ml-platform
```

The application will be available at:

- API: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health

---

## Development

Install dependencies:

```bash
make install
```

Run the application:

```bash
make run
```

Install the Git hooks (run once):

```bash
uv run pre-commit install
```

Run all pre-commit hooks manually:

```bash
make precommit
```
