# Wrapper to expose the app at the root level for agentcore CLI
from src.infrastructure.api import app

__all__ = ["app"]

if __name__ == "__main__":
    app.run()

# Look for this Claude Chat for more information: https://claude.ai/chat/6b6bccd0-a772-49d7-a412-1eb9b0e534ca