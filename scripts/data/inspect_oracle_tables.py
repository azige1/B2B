import ast
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import oracledb


DEFAULT_TABLES = [
    "BOSNDS3.V_IRS_ORDERFTP",
    "BOSNDS3.V_IRS_PRODUCT",
    "BOSNDS3.V_IRS_STORAGE",
    "BOSNDS3.V_IRS_B2BSTORAGE",
    "BOSNDS3.V_IRS_EVENT",
    "BOSNDS3.V_IRS_PREORDER",
    "BOSNDS3.V_IRS_PRO_DATA",
    "BOSNDS3.V_IRS_STORE",
    "BOSNDS3.V_IRS_CUS_PROFILE",
]


def load_get_store_config(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"LIB_DIR", "DB_CONFIG"}:
                values[target.id] = ast.literal_eval(node.value)
    if "LIB_DIR" not in values or "DB_CONFIG" not in values:
        raise ValueError(f"Could not read LIB_DIR/DB_CONFIG from {path}")
    return values["LIB_DIR"], values["DB_CONFIG"]


def json_value(value):
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def inspect_table(connection, table):
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = int(cursor.fetchone()[0])

        cursor.execute(f"SELECT * FROM {table} WHERE ROWNUM <= 3")
        columns = [
            {
                "name": item[0],
                "type": str(item[1]),
                "display_size": item[2],
                "internal_size": item[3],
                "precision": item[4],
                "scale": item[5],
                "nullable": item[6],
            }
            for item in cursor.description
        ]
        samples = [
            {
                column["name"]: json_value(value)
                for column, value in zip(columns, row)
            }
            for row in cursor.fetchall()
        ]
    return {
        "table": table,
        "row_count": row_count,
        "columns": columns,
        "sample_rows": samples,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/root/get_store.py"
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/oracle_table_inventory.json")
    tables = sys.argv[3:] or DEFAULT_TABLES

    library_dir, config = load_get_store_config(config_path)
    oracledb.init_oracle_client(lib_dir=library_dir)
    dsn = oracledb.makedsn(
        config["host"],
        config["port"],
        sid=config["sid"],
    )
    connection = oracledb.connect(
        user=config["user"],
        password=config["password"],
        dsn=dsn,
    )
    results = []
    try:
        for table in tables:
            try:
                result = inspect_table(connection, table)
                result["status"] = "ok"
            except Exception as exc:
                result = {
                    "table": table,
                    "status": "error",
                    "error": str(exc),
                }
            results.append(result)
            print(f"{table}: {result['status']} rows={result.get('row_count', '-')}")
    finally:
        connection.close()

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] {output_path}")


if __name__ == "__main__":
    main()
