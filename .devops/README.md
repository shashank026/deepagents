# DeepAgents deployment

DeepAgents has no published host port. The backend reaches it at
`http://deepagents:8010` through the external `tracex-network`.

```bash
docker network create tracex-network 2>/dev/null || true
cd deepagents/.devops
cp .env.example .env
chmod 600 .env
# Set GOOGLE_API_KEY, then:
docker compose up -d --build
docker compose ps
```

