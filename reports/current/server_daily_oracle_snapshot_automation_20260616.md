# Server Daily Oracle Snapshot Automation (2026-06-16)

## 结论

已在新服务器上配置每日自动取数任务。服务器每天上海时间 `03:30` 从甲方 Oracle 源库导出一份完整 CSV 快照，并生成日志、manifest、压缩包和 SHA-256 校验文件。

该任务只负责把数据稳定拉到服务器，不会自动覆盖本地 `data_warehouse`，也不会自动重训模型。

## 服务器配置

- 服务器：`root@121.40.254.36`
- 时区：`Asia/Shanghai`
- cron 状态：`active`
- 自动任务：

```cron
30 3 * * * /root/run_daily_oracle_snapshot.sh >> /root/client_data_snapshots/logs/cron_daily_snapshot.log 2>&1
```

## 服务器文件

| 用途 | 路径 |
| --- | --- |
| 自动导出脚本 | `/root/run_daily_oracle_snapshot.sh` |
| cron 模板 | `/root/b2b_daily_oracle_snapshot.cron` |
| 导出 Python 脚本 | `/root/export_oracle_snapshot.py` |
| 表审计 Python 脚本 | `/root/inspect_oracle_tables.py` |
| Oracle 配置来源 | `/root/get_store.py` |
| Oracle Python 环境 | `/root/oracle-env/bin/python` |
| Oracle Instant Client | `/root/instantclient_19_8` |
| 快照根目录 | `/root/client_data_snapshots` |
| 日志目录 | `/root/client_data_snapshots/logs` |

## 每日输出

以 `YYYYMMDD` 为日期标签，每天会生成：

```text
/root/client_data_snapshots/client_snapshot_YYYYMMDD/
/root/client_data_snapshots/client_snapshot_YYYYMMDD.tar.gz
/root/client_data_snapshots/client_snapshot_YYYYMMDD.tar.gz.sha256
/root/client_data_snapshots/oracle_table_inventory_YYYYMMDD.json
/root/client_data_snapshots/logs/daily_oracle_snapshot_YYYYMMDD.log
```

快照目录中包含：

- `V_IRS_ORDERFTP.csv`
- `V_IRS_PRODUCT.csv`
- `V_IRS_STORAGE.csv`
- `V_IRS_B2BSTORAGE.csv`
- `V_IRS_EVENT.csv`
- `V_IRS_STORE.csv`
- `V_IRS_CUS_PROFILE.csv`
- `V_IRS_PREORDER.csv`
- `V_IRS_ORDER.csv`
- `manifest.json`
- `oracle_table_inventory_YYYYMMDD.json`
- `_SUCCESS`

## 安全机制

- 使用 `/tmp/b2b_oracle_snapshot.lock` 加锁，避免重复任务并发导出。
- 如果同一天快照目录已有 `_SUCCESS`，默认跳过，避免重复覆盖。
- 设置 `FORCE=1` 可强制重跑同一天。
- 设置 `DRY_RUN=1` 可只做环境检查，不连接导出。
- 压缩包导出后会生成 `.sha256` 文件，便于本地下载后校验。

## 手工命令

查看定时任务：

```bash
crontab -l
```

手工 dry-run：

```bash
DRY_RUN=1 /root/run_daily_oracle_snapshot.sh 20990101
```

手工导出指定日期标签：

```bash
/root/run_daily_oracle_snapshot.sh 20260617
```

强制重跑指定日期：

```bash
FORCE=1 /root/run_daily_oracle_snapshot.sh 20260617
```

查看日志：

```bash
tail -100 /root/client_data_snapshots/logs/daily_oracle_snapshot_20260617.log
tail -100 /root/client_data_snapshots/logs/cron_daily_snapshot.log
```

下载到本地示例：

```powershell
New-Item -ItemType Directory -Force .\data\incoming\server_20260617 | Out-Null
& 'C:\Windows\System32\OpenSSH\scp.exe' -i C:\Users\Hendo\.ssh\id_rsa root@121.40.254.36:/root/client_data_snapshots/client_snapshot_20260617.tar.gz .\data\incoming\server_20260617\
& 'C:\Windows\System32\OpenSSH\scp.exe' -i C:\Users\Hendo\.ssh\id_rsa root@121.40.254.36:/root/client_data_snapshots/client_snapshot_20260617.tar.gz.sha256 .\data\incoming\server_20260617\
```

本地校验：

```powershell
Get-FileHash .\data\incoming\server_20260617\client_snapshot_20260617.tar.gz -Algorithm SHA256
Get-Content .\data\incoming\server_20260617\client_snapshot_20260617.tar.gz.sha256
```

## 当前验证结果

- `bash -n /root/run_daily_oracle_snapshot.sh`：通过。
- `DRY_RUN=1 /root/run_daily_oracle_snapshot.sh 20990101`：通过。
- `cron` 服务：`active`。
- 当前 root crontab 已包含每日 `03:30` 任务。

## 注意事项

- 该导出是逐表实时导出，不是数据库级一致性快照。库在导出期间继续写入时，不同表之间可能有轻微时间差。
- `V_IRS_ORDERFTP` 仍是正式标签源；`V_IRS_ORDER` 只作为备份，不用于训练标签。
- 自动任务不会清理旧快照。后续如果磁盘空间紧张，再加明确的保留策略，不要静默删除历史数据。
