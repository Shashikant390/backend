# alembic/env.py
from logging.config import fileConfig
import os
import sys
from sqlalchemy import engine_from_config, pool
from alembic import context

config = context.config
fileConfig(config.config_file_name)

# prefer env var if set (override alembic.ini)
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    # escape % for configparser
    config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))

# Ensure project root is on sys.path so package imports work when running alembic from anywhere.
# Adjust '..' if alembic folder is located elsewhere relative to package root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- import your project metadata ----
# Try to import the app package first (preferred)
try:
    # Make sure your package is named `sh_app` and contains db.py and models.py
    from sh_app.db import engine as app_engine  # optional
    from sh_app.models import Base
    target_metadata = Base.metadata
except Exception as e:
    # Fallback: try relative imports (if running inside package)
    try:
        from db import engine as app_engine
        from models import Base
        target_metadata = Base.metadata
    except Exception:
        # If both fail, set target_metadata to None — autogenerate won't work but migrations can still run.
        target_metadata = None
        # Optionally print helpful error for debugging
        print("Warning: couldn't import project models (sh_app.models.Base). Autogenerate will be disabled.", e)

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
