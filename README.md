# GeneX AI: General Computational Biology Framework
# (GeneX AI：通用计算生物学 AI 框架)

[English](#english) | [中文](#chinese)

---
<a name="english"></a>
## 🧬 About GeneX AI
**GeneX AI (formerly XenoGenesis)** is a general-purpose computational biology and AI framework. Moving beyond simple LLM wrappers, it deeply integrates deterministic physical, chemical, and biological solvers to assist researchers in gene function prediction, CRISPR design, toxicity evaluation, and multi-agent systems biology debates. 

As an **extended feature**, it inherently supports advanced Xenobiology and Terraforming simulations (e.g., simulating Silicon-based life forms and geological-timescale ecological ODEs).

### Core Features
- **General Biological Analysis**: Precision deterministic proxies for CRISPR off-target scoring, HLA immunogenicity, and toxicity prediction.
- **Multi-Agent Debate Protocol**: Built-in `Architect -> Reviewer -> Experimentalist` autonomous debate loop to ensure scientific rigor.
- **Real-World DB Gateway**: Native integration with Ensembl, UniProt, and DepMap for retrieving ground-truth evidence (Level A ~ Level C grading).
- **Hardcore Physics Solvers (Bonus)**: Thermodynamics solver (Gibbs $\Delta G$), 10,000-year timescale Lotka-Volterra ODEs using LSODA, and non-natural nucleic acid evaluation.
- **Lab Automation**: Direct compilation of Opentrons OT-2 Python scripts.

### Database Architecture (数据库架构)
- **Lazy-Loading Cache (懒加载动态缓存)**: By default, the system comes with a lightweight "Seed DB" for offline testing. When you query any unknown gene (out of the 20,000+ human genes), the `GlobalBioDataGateway` will fetch it from Ensembl/UniProt in real-time and permanently cache it in your local SQLite DB. Your database grows as you research.
- **Full Genome Dump (全量离线包)**: For enterprise users who need strict offline air-gapped environments, the full 20k+ Human Genome SQLite dump will be provided in the GitHub Releases page.
```bash
# Windows
install.bat

# Mac/Linux
bash install.sh
```

### License & Commercial Use
This open-source version is licensed under the strict **GNU Affero General Public License v3.0 (AGPL-3.0)**. 
- Any use of this code over a network (e.g., as a SaaS, API, or Web service) **requires** you to open-source your entire backend codebase.
- **Commercial License**: If you wish to deploy this in closed-source commercial products, internal enterprise systems, or private clouds, please contact the author for a commercial license.
- **Contact**: [2141595982@qq.com](mailto:2141595982@qq.com)

---
<a name="chinese"></a>
## 🧬 关于 GeneX AI
**GeneX AI（原名 XenoGenesis）** 是一个**通用型计算生物学 AI 框架**。它不仅仅是大模型的简单套壳，而是深度集成了确定性的物理、化学和生物学求解器，以协助研究人员进行基因功能预测、CRISPR 设计、毒性评估以及多智能体系统生物学辩论。

作为 **扩展能力**，它天生支持高级异种生物学（Xenobiology）和行星地球化（Terraforming）模拟（例如：模拟硅基生命和万年地质尺度的生态微分方程）。

### 核心特性
- **通用生物学分析**：针对 CRISPR 脱靶打分、HLA 免疫原性和毒性预测的精确确定性算法（向药审标准看齐）。
- **多 Agent 辩论协议**：内置 `架构师 (Architect) -> 审查员 (Reviewer) -> 实验员 (Experimentalist)` 自主辩论循环，确保科学严谨性。
- **真实数据库网关**：原生集成 Ensembl, UniProt, 和 DepMap 提取真实证据（支持 Level A ~ Level C 证据分级）。
- **硬核物理求解器 (附带卖点)**：热力学求解器（吉布斯自由能 $\Delta G$）、支持万年尺度演化的 Lotka-Volterra 常微分方程（LSODA算法），以及非天然核酸组装评估。
- **云实验室自动化**：直接编译生成 Opentrons OT-2 机械臂液体处理 Python 脚本。

### 数据库架构策略
- **懒加载动态缓存 (Lazy-Loading Cache)**：为保证开源包的轻量级，系统默认仅附带用于离线测试的“基础种子库”。当您查询任何未收录的基因时，系统底层网关会瞬间从云端（Ensembl）抓取该基因的真实注释，并**永久存入本地 SQLite**。随着您的使用，本地库将自动生长为您的专属全量库。
- **全量离线数据包 (The 20k Base Dump)**：对于需要绝对物理隔离（Air-gapped）的药企级部署，我们将在 GitHub Releases 页面提供完整的人类 20,000+ 蛋白编码基因的 SQLite 数据包供独立下载替换。

### 快速开始
```bash
# Windows
install.bat

# Mac/Linux
bash install.sh
```

### 许可证与商业授权
本项目开源版本采用严格的 **GNU Affero General Public License v3.0 (AGPL-3.0)** 协议。
- 任何通过网络（如作为 SaaS、API、Web 服务）使用本代码的行为，**必须**开源其服务端的所有相关源码。
- **商业授权 (Commercial License)**：如果您希望在闭源的商业产品、企业内部系统或私有云中部署使用，请联系作者获取商业授权。
- **商务联系邮箱**：[2141595982@qq.com](mailto:2141595982@qq.com)

### 免责声明
本项目生成的基因组蓝图、病原体突变推演及自动化合成脚本仅供学术研究、计算生物学推演使用。请严格遵守所在国家和地区的合成生物学安全法规与伦理审查要求。