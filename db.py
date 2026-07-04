"""Query helper for the devp-read RDS read replica (read-only).

Usage: python3 db.py [database] "SQL"
Credentials come from Secrets Manager at runtime; never written to disk.
"""
import json
import subprocess
import sys

import pg8000.native

READ_REPLICA_HOST = "devp-read.czyaggyi6bp9.ap-south-1.rds.amazonaws.com"


def get_creds():
    out = subprocess.run(
        ["aws", "secretsmanager", "get-secret-value", "--secret-id", "rdsDevProd",
         "--query", "SecretString", "--output", "text"],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def connect(database="postgres"):
    creds = get_creds()
    con = pg8000.native.Connection(
        user=creds["username"], password=creds["password"],
        host=READ_REPLICA_HOST, port=int(creds["port"]),
        database=database, timeout=30,
    )
    con.run("SET default_transaction_read_only = on")
    con.run("SET statement_timeout = '120s'")
    return con


def main():
    database = sys.argv[1] if len(sys.argv) > 2 else "postgres"
    sql = sys.argv[-1]
    con = connect(database)
    try:
        rows = con.run(sql)
        cols = [c["name"] for c in con.columns] if con.columns else []
        print("\t".join(cols))
        for r in rows or []:
            print("\t".join("" if v is None else str(v)[:300] for v in r))
    finally:
        con.close()


if __name__ == "__main__":
    main()
