# B2B Replenishment System

B2B 鏈嶈琛ヨ揣棰勬祴椤圭洰銆傚綋鍓嶄粨搴撳凡缁忎粠鏃╂湡鐨?LSTM 涓荤嚎锛屾紨杩涘埌浠ユ爲妯″瀷涓烘寮忎富绾跨殑鐮旂┒涓庝氦浠樺伐浣滃尯銆?
## 褰撳墠鐘舵€?
- 褰撳墠瀹樻柟闃舵锛歚phase7`
- 褰撳墠瀹樻柟鐘舵€侊細`frozen`
- 褰撳墠瀹樻柟涓荤嚎锛歚tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- 褰撳墠瀹樻柟鏍戞ā鍨嬪鏃忥細`LightGBM`
- 褰撳墠 phase8 鏂瑰悜锛歚event + inventory`锛屼絾浠嶅浜?shadow / exploratory 闃舵

褰撳墠姝ｅ紡鍏ュ彛浼樺厛鐪嬶細

- `PROJECT_INDEX.md`
- `reports/current/current_mainline.json`
- `reports/current/current_freeze_summary.md`
- `reports/current/current_model_compare_summary.md`
- `reports/current/phase8_direction_note.md`

## 浠撳簱缁撴瀯

```text
B2B_Replenishment_System/
鈹溾攢鈹€ src/                 # 鏍稿績浠ｇ爜锛欵TL銆佺壒寰併€佽缁冦€佽瘎浼般€佹帹鐞嗐€佸垎鏋?鈹溾攢鈹€ config/              # 妯″瀷涓庢暟鎹厤缃?鈹溾攢鈹€ docs/                # 椤圭洰璇存槑涓庡伐浣滆鑼?鈹溾攢鈹€ reports/             # 褰撳墠缁撹銆侀樁娈垫€荤粨銆佸巻鍙插疄楠岀粨璁?鈹溾攢鈹€ scripts/             # 璇婃柇鑴氭湰涓庡巻鍙?runner
鈹溾攢鈹€ data/                # 鏁版嵁璧勪骇绱㈠紩涓庢湰鍦板疄楠岃祫浜ф槧灏?鈹溾攢鈹€ data_warehouse/      # 鍘熷鎶藉彇涓庡揩鐓ф暟鎹紙榛樿涓嶇撼鍏?Git锛?鈹斺攢鈹€ models/              # 鏈湴妯″瀷浜х墿锛堥粯璁や笉绾冲叆 Git锛?```

## 甯哥敤鍏ュ彛

褰撳墠椤圭洰鏇村亸鈥滈樁娈?runner + 鎶ュ憡褰掓。鈥濈殑宸ヤ綔鏂瑰紡锛屽父鐢ㄥ叆鍙ｅ寘鎷細

```bash
# 鏌ョ湅褰撳墠瀹樻柟鐘舵€?
python scripts/runners/phase7/run_phase7_freeze.py

# 鐢熸垚褰撳墠瀹樻柟 December compare
python scripts/runners/phase7/run_phase7i_full_model_compare.py

# phase8 鍑嗗鍒嗘瀽
python scripts/runners/phase8/run_phase8a_prep.py

# phase8 搴撳瓨绾︽潫鍒嗘瀽
python scripts/runners/phase8/run_phase8f_inventory_constraint_pack.py

# 浠撳簱鍗敓妫€鏌?
python scripts/diagnostic/check_git_hygiene.py
```

濡傛灉鏄槄璇昏€屼笉鏄噸璺戯紝浼樺厛鐩存帴鐪?`reports/current/`銆?
## 鏁版嵁涓庣増鏈鐞?
- Git 涓昏璺熻釜婧愮爜銆侀厤缃€佸叧閿枃妗ｃ€侀樁娈电粨璁恒€?- 鍘熷鏁版嵁銆佸鐞嗕骇鐗┿€佹ā鍨嬫潈閲嶃€佸鍑哄瀷鎶ヨ〃榛樿涓嶇撼鍏ョ増鏈簱銆?- Git 瑙勫垯瑙?`docs/GIT_WORKFLOW.md`銆?
## 褰撳墠宸ヤ綔閲嶇偣

- 淇濇寔 `phase7` 瀹樻柟涓荤嚎绋冲畾鍙拷婧?- 鍩轰簬 `event + inventory` 鍋?phase8 鏂瑰悜楠岃瘉
- 绛夊緟瀹㈡埛纭 `V_IRS_ORDERFTP` 璇箟涓庣敓鍛藉懆鏈熻〃鍚庯紝鍐嶈繘鍏ユ寮?phase8


