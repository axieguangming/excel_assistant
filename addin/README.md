# Excel AI 助手 — Office 加载项（方案1）

把智能体能力嵌入**真实 Excel**：在 Excel 里用自然语言分析当前工作表，结果写入**新工作表**，并生成 **Excel 原生图表**（可编辑、可改样式）。

> 目标环境：**Excel 2021 标准版（Windows）**，使用传统 `v1.1` manifest，兼容性最好。
> 后端（FastAPI + LangGraph 智能体）保持不变，加载项只是新前端。

---

## 0. 原理与文件
- `static/addin/taskpane.html` — 任务窗格 UI（聊天 + 读写 Excel），由后端在同一 HTTPS 源下托管：`https://localhost:8443/addin/taskpane.html`
- `addin/manifest.xml` — Excel 加载项清单（侧加载用，从磁盘打开，不经服务器）
- `addin/make_cert.sh` — 生成 localhost 自签名证书（Excel 强制要求 HTTPS）
- 新增后端接口：`POST /api/session` — 用「当前工作表」的数据（表头+数据行）建立会话，无需上传文件
- 结果回写：taskpane 用 Office.js 新建工作表、写入数据、`charts.add` 画原生柱状图；若 AI 也生成了 matplotlib 图，会以图片形式一并插入

---

## 1. 生成并信任 HTTPS 证书（一次性）
Excel 加载项的页面必须是 **HTTPS**。本地开发用自签名证书：

```bash
# Git Bash，在项目根目录
bash addin/make_cert.sh
```

生成后，用**管理员 PowerShell**信任该证书（否则 Excel 打不开）：

```powershell
certutil -addstore -f ROOT addin\certs\cert.pem
```

> 验证：浏览器访问 `https://localhost:8443` 显示“安全/无警告”即成功（证书被系统信任）。

---

## 2. 以 HTTPS 启动后端
在 **conda 环境 excel_assistant** 中：

```bash
conda activate excel_assistant
export SSL_CERTFILE=addin/certs/cert.pem
export SSL_KEYFILE=addin/certs/key.pem
python app.py          # 默认 https://0.0.0.0:8443
```

（不设 SSL 变量会退回 HTTP，加载项将无法加载，仅普通网页版可用。）

---

## 3. 侧加载到 Excel 2021（Windows）
Excel 2021 桌面版用「受信任的目录 / 共享文件夹」方式侧加载：

1. 新建一个存放清单的目录，例如 `C:\ExcelAIAddin`，把 **`addin/manifest.xml`** 复制进去。
2. 打开 Excel → `文件` → `选项` → `信任中心` → `信任中心设置` → `受信任的目录`。
   - `新增目录`：填 `C:\ExcelAIAddin`，勾选「**询问是否添加加载项**」(Warn before installing) 或「在我的目录中显示」。
3. `插入` 选项卡 → `我的加载项`（或 `加载项`）→ 左侧选「**受信任的目录 / 共享文件夹**」→ 选择 **Excel AI 助手** → `确定`。
4. 首次加载会提示，确认信任即可。之后主页(Ribbon)「**Excel AI**」组里会出现「**AI 助手**」按钮。

> 若菜单里没有「我的加载项」，确认 Excel 为 Microsoft 365/2021 且已登录；或用命令 `excel.exe /sharelist C:\ExcelAIAddin`。

---

## 4. 使用
1. 在任意工作表放一份**带表头**的数据（第一行是列名）。
2. 点 Ribbon 上的「AI 助手」打开任务窗格 → 自动「连接工作表」（或点右上「重新连接」）。
3. 提问，例如：
   - `各分类求和并画柱状图`
   - `数值最大前10行`
   - `占比分布饼图`
4. 结果会**新建一个工作表**写入数据，并尝试生成 **Excel 原生柱状图**；聊天气泡里有「定位到工作表」按钮。
   - 想改图表类型/配色：直接点中 Excel 里的图表用原生功能调整。

---

## 5. 常见问题
- **加载项列表为空/报错**：多半是证书没被系统信任，或后端未以 HTTPS 启动。回到第 1、2 步。
- **页面能开但“无法连接”**：确认 `https://localhost:8443/addin/taskpane.html` 浏览器可直接访问。
- **没生成原生图表**：结果表需 ≥2 列且含至少 1 个数值列；否则只写数据表。matplotlib 图仍会作为图片插入。
- **公司网络/代理**：Office.js 从微软 CDN 加载，需允许访问 `appsforoffice.microsoft.com`。
- **换端口/主机名**：改 `addin/manifest.xml` 里所有 `https://localhost:8443` 为对应地址，并让证书 CN/SAN 匹配。
