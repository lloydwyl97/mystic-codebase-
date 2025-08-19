import os
try:
    if os.getenv("NO_IPV6","1")=="1":
        import urllib3.util.connection as conn
        conn.HAS_IPV6 = False
except Exception:
    pass
