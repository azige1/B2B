import oracledb
import os
import sys
import yaml

class OracleConnector:
    def __init__(self, config_path="./config/data_config.yaml"):
        # 1. 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.db_cfg = self.config['database']
        
        # 2. 初始化客户端 (Thick Mode)
        self._init_client()

    def _init_client(self):
        """初始化 Oracle 客户端 (激活 Thick Mode)"""
        lib_dir = self.db_cfg.get('lib_dir')
        
        # 强校验：如果路径不对，直接报错，不再掩耳盗铃
        if not lib_dir or not os.path.exists(lib_dir):
            print(f"❌ [致命错误] Oracle 客户端路径不存在: {lib_dir}")
            print("   请下载 Instant Client 并检查 config/data_config.yaml")
            sys.exit(1)

        try:
            oracledb.init_oracle_client(lib_dir=lib_dir)
            # print("✅ Oracle Thick Mode 已激活")
        except oracledb.DatabaseError as e:
            # 如果程序多次运行，会报 'already initialized'，这是正常的
            if "already been initialized" not in str(e):
                print(f"❌ Oracle Client 加载失败: {e}")
                sys.exit(1)

    def get_connection(self):
        """获取数据库连接对象"""
        try:
            conn = oracledb.connect(
                user=self.db_cfg['user'],
                password=self.db_cfg['password'],
                dsn=self.db_cfg['dsn']
            )
            return conn
        except Exception as e:
            print(f"❌ 无法连接到数据库: {self.db_cfg['dsn']}")
            print(f"   错误详情: {e}")
            raise e