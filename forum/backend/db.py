from sqlalchemy import create_engine, text


def one(conn, sql, **params):
    return conn.execute(text(sql), params).mappings().first()


def all_rows(conn, sql, **params):
    return list(conn.execute(text(sql), params).mappings())


def execute(conn, sql, **params):
    return conn.execute(text(sql), params)


def make_engine(url):
    return create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True,
                         pool_timeout=10, hide_parameters=True,
                         connect_args={"connect_timeout": 5,
                                       "options": "-c statement_timeout=10000 -c lock_timeout=5000"})
