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

## 1. HTTPS 证书（自动配置）
Excel 加载项的页面必须是 **HTTPS**。Windows 下首次运行 `python app.py` 时，
程序会自动生成 localhost 自签名证书并加入当前 Windows 用户的受信任根证书存储；
后续启动会复用 `addin/certs/cert.pem` 和 `addin/certs/key.pem`，无需重复生成或手动信任。

证书私钥保存在本机项目目录，请勿分享或提交。若证书文件被删除或过期，下次启动会重新生成并再次信任。

---

## 2. 启动后端
在项目根目录的 PowerShell 中运行：

```powershell
python app.py
```

程序默认仅监听本机并以 HTTPS 提供服务：`https://localhost:8443`。如果需要使用已有证书，
可同时设置 `SSL_CERTFILE` 和 `SSL_KEYFILE` 覆盖自动配置；不要将未鉴权服务直接暴露到局域网或公网。

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
